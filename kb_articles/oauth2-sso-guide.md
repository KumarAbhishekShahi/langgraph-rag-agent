# OAuth2 Single Sign-On (SSO) Integration Guide

## Overview
This guide covers integrating OAuth2 SSO using Azure AD, Okta, or Keycloak
into enterprise Spring Boot applications.

## Supported Flows
- Authorization Code Flow with PKCE (recommended for web apps)
- Client Credentials Flow (service-to-service)
- Device Authorization Flow (CLI tools)

## Spring Boot Configuration
```yaml
spring:
  security:
    oauth2:
      client:
        registration:
          azure:
            client-id: ${AZURE_CLIENT_ID}
            client-secret: ${AZURE_CLIENT_SECRET}
            scope: openid, profile, email
            redirect-uri: "{baseUrl}/login/oauth2/code/{registrationId}"
        provider:
          azure:
            issuer-uri: https://login.microsoftonline.com/{tenant-id}/v2.0
```

## Token Management
- Access tokens expire in 1 hour (default)
- Refresh tokens valid for 90 days
- Use `TokenStore` bean for distributed session management
- Store tokens in Redis for horizontal scaling

## Security Checklist
- [ ] Enable HTTPS only (redirect HTTP → HTTPS)
- [ ] Validate `iss` and `aud` claims in JWT
- [ ] Implement CSRF protection
- [ ] Set `SameSite=Strict` on session cookies
- [ ] Auto-logout after 8 hours of inactivity

## Acceptance Criteria
- User can log in with company Microsoft/Google account
- Session expires after 8 hours of inactivity
- Failed login attempts are logged to audit trail
- MFA enforcement if configured in identity provider
