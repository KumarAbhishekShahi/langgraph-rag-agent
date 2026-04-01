# Payment Gateway Integration Standards

## Supported Gateways
- Stripe (primary — card, UPI, wallets)
- Razorpay (India domestic)
- PayPal (international)

## PCI-DSS Compliance Requirements
- Never store raw card numbers (PAN) — use tokenization
- Transmit card data only over TLS 1.2+
- Log all payment events to immutable audit log
- Quarterly vulnerability scans required

## Stripe Integration Pattern (Spring Boot)
```java
@Service
public class PaymentService {
    private final Stripe stripe;

    public PaymentIntent createPaymentIntent(long amountPaise, String currency) {
        PaymentIntentCreateParams params = PaymentIntentCreateParams.builder()
            .setAmount(amountPaise)
            .setCurrency(currency)
            .setAutomaticPaymentMethods(
                PaymentIntentCreateParams.AutomaticPaymentMethods.builder()
                    .setEnabled(true).build())
            .build();
        return PaymentIntent.create(params);
    }
}
```

## Webhook Handling
- Verify Stripe-Signature header on every webhook
- Idempotency key required for retry safety
- Process webhooks asynchronously (use @Async or Kafka)

## Refund Policy Rules
- Full refund within 7 days: automatic approval
- Partial refund within 30 days: requires manager approval
- After 30 days: escalate to finance team

## Error Codes
| Code | Meaning | Action |
|------|---------|--------|
| card_declined | Card refused by issuer | Ask user to retry or use another card |
| insufficient_funds | Low balance | Inform user |
| expired_card | Card expired | Ask for updated card |
