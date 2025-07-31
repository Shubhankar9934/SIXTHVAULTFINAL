import { NextResponse } from 'next/server'
import type { NextRequest } from 'next/server'

// Route categorization for different access levels
const PUBLIC_ROUTES = [
  '/',
  '/login',
  '/register', 
  '/forgot-password',
  '/privacy-policy',
  '/terms-of-service'
]

const VERIFICATION_ROUTES = [
  '/verify',
  '/verify-email',
  '/verify-phone'
]

const PROTECTED_ROUTES = {
  user: [
    '/vault',
    '/documents',
    '/upload',
    '/profile',
    '/settings',
    '/dashboard'
  ],
  admin: [
    '/admin',
    '/admin/users',
    '/admin/analytics',
    '/admin/settings'
  ],
  manager: [
    '/manage',
    '/manage/team',
    '/manage/reports'
  ]
}

type ConditionalRouteKey = 'reset-password' | 'accept-invite' | 'team-join'
type ConditionalRoutesConfig = {
  [K in ConditionalRouteKey]: {
    requiredParams: string[]
    redirect: string
  }
}

const CONDITIONAL_ROUTES: ConditionalRoutesConfig = {
  'reset-password': {
    requiredParams: ['token'],
    redirect: '/login'
  },
  'accept-invite': {
    requiredParams: ['token', 'team'],
    redirect: '/login'
  },
  'team-join': {
    requiredParams: ['id', 'invite'],
    redirect: '/dashboard'
  }
}

// Define roles and their hierarchies
const ROLE_HIERARCHY = {
  admin: ['admin', 'manager', 'user'],
  manager: ['manager', 'user'],
  user: ['user']
}

// Route types and conditions
type RouteType = 'public' | 'protected' | 'verification' | 'conditional'
type UserRole = 'admin' | 'manager' | 'user'

interface ConditionalRouteConfig {
  requiredParams: string[]
  redirect: string
}

interface TokenPayload {
  sub: string
  role: UserRole
  exp: number
  iat: number
}

type RouteCondition = {
  check: () => boolean
  redirect: string
}

type RouteConditions = {
  [key: string]: RouteCondition
}

// Session states for different authentication levels
enum SessionState {
  AUTHENTICATED = 'authenticated',
  PENDING_VERIFICATION = 'pending_verification',
  PASSWORD_RESET = 'password_reset',
  PUBLIC = 'public'
}

// Error types for authentication failures
enum AuthError {
  TOKEN_EXPIRED = 'token_expired',
  INVALID_SESSION = 'invalid_session',
  INSUFFICIENT_PERMISSIONS = 'insufficient_permissions',
  RATE_LIMITED = 'rate_limited',
  SECURITY_VIOLATION = 'security_violation'
}

export function middleware(request: NextRequest) {
  const response = NextResponse.next();
  response.headers.set('ngrok-skip-browser-warning', 'true');
  response.headers.set('User-Agent', 'CustomAgent/1.0');

  const { pathname } = request.nextUrl
  
  // Helper functions for redirects
  const redirectToLogin = () => {
    const url = new URL('/login', request.url)
    if (!pathname.startsWith('/login')) {
      url.searchParams.set('redirect', pathname)
    }
    return NextResponse.redirect(url)
  }

  const redirectToAppropriate = (route: string) => {
    const url = new URL(route, request.url)
    return NextResponse.redirect(url)
  }

  // Helper function to check role access
  const hasRoleAccess = (userRole: UserRole, requiredRole: UserRole): boolean => {
    // A user can access routes for their role and any roles they have access to
    const userHierarchy = ROLE_HIERARCHY[userRole]
    const hasAccess = userHierarchy?.includes(requiredRole) || false
    console.log(`Middleware: Role check - User: ${userRole}, Required: ${requiredRole}, Hierarchy: [${userHierarchy?.join(', ')}], Access: ${hasAccess}`)
    return hasAccess
  }

  // Helper function to validate token
  const validateToken = (token: string): TokenPayload | null => {
    try {
      // Basic JWT structure validation
      const parts = token.split('.')
      if (parts.length !== 3) {
        console.log('Middleware: Invalid token structure')
        return null
      }

      // Decode the payload (we can't verify signature in middleware without the secret)
      const payload = JSON.parse(atob(parts[1])) as TokenPayload
      
      // Check if token is expired
      if (Date.now() >= payload.exp * 1000) {
        console.log('Middleware: Token expired')
        return null
      }
      
      // Basic structure validation
      if (!payload.sub || !payload.role || !payload.exp || !payload.iat) {
        console.log('Middleware: Token missing required fields')
        return null
      }
      
      console.log(`Middleware: Token validation successful for user: ${payload.sub}, role: ${payload.role}`)
      return payload
    } catch (error) {
      console.log('Middleware: Token validation error:', error)
      return null
    }
  }

  // Check route type
  const routeType: RouteType = (() => {
    if (PUBLIC_ROUTES.some(route => pathname.startsWith(route))) return 'public'
    if (VERIFICATION_ROUTES.some(route => pathname.startsWith(route))) return 'verification'
    if (Object.values(PROTECTED_ROUTES).flat().some(route => pathname.startsWith(route))) return 'protected'
    if (Object.keys(CONDITIONAL_ROUTES).some(route => pathname.startsWith(`/${route}`))) return 'conditional'
    return 'protected' // Default to protected
  })()

  // Get and validate authentication token
  const token = request.cookies.get('auth-token')?.value
  const tokenPayload = token ? validateToken(token) : null

  // Following the flowchart logic
  switch (routeType) {
    case 'public':
      // Always allow access to public routes
      return response

    case 'protected':
      // Check auth token and role access
      if (!tokenPayload) {
        return redirectToLogin()
      }

      try {
        // Check if user has access to the requested route
        const routeRoleEntry = Object.entries(PROTECTED_ROUTES).find(([_, routes]) =>
          routes.some(route => pathname.startsWith(route))
        )

        if (!routeRoleEntry) {
          // Route not found in protected routes, deny access
          return redirectToLogin()
        }

        const requiredRole = routeRoleEntry[0] as UserRole
        
        // Check if user's role has access to the required role level
        // A user can access routes for their role and lower privilege roles
        if (!hasRoleAccess(tokenPayload.role, requiredRole)) {
          console.log(`Middleware: Access denied. User role: ${tokenPayload.role}, Required: ${requiredRole}`)
          return redirectToLogin()
        }

        console.log(`Middleware: Access granted. User role: ${tokenPayload.role}, Required: ${requiredRole}`)
        return response
      } catch (error) {
        console.log('Middleware: Error in role validation:', error)
        return redirectToLogin()
      }

    case 'verification':
      // For verification routes, only check that at least one param exists
      // This allows both email/code input and email/token verification flows
      const hasEmailParam = request.nextUrl.searchParams.has('email')
      if (!hasEmailParam) {
        return redirectToLogin()
      }

      // Check if token is expired (you can add more validation here)
      try {
        // Verify token validity
        // You can add token expiration check here
        return response
      } catch {
        return redirectToLogin()
      }

    case 'conditional':
      // Get route configuration
      const routeKey = Object.keys(CONDITIONAL_ROUTES).find(route => 
        pathname.startsWith(`/${route}`)
      ) as ConditionalRouteKey | undefined

      if (routeKey) {
        const config = CONDITIONAL_ROUTES[routeKey]
        
        // Check if all required parameters are present
        const missingParams = config.requiredParams.filter(
          (param: string) => !request.nextUrl.searchParams.get(param)
        )

        if (missingParams.length > 0) {
          return redirectToAppropriate(config.redirect)
        }

        // Additional validation can be added here
        // For example, validate token format, check expiration, etc.
      }
      return response

    default:
      // Default to protected route behavior
      if (!token) {
        return redirectToLogin()
      }
      return response
  }
}

export const config = {
  matcher: [
    /*
     * Match all request paths except for the ones starting with:
     * - api (API routes)
     * - _next/static (static files)
     * - _next/image (image optimization files)
     * - favicon.ico (favicon file)
     */
    '/((?!api|_next/static|_next/image|favicon.ico).*)',
  ],
}
