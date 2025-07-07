from django.db import models

class Transaction(models.Model):
    SENDER_CHOICES = [
      # optional if you ever want more than one constant
      ('MyBankBackend', 'MyBankBackend'),
    ]

    sender      = models.CharField(max_length=100, choices=SENDER_CHOICES, default='MyBankBackend')
    receiver    = models.CharField(max_length=100)
    value       = models.DecimalField(max_digits=10, decimal_places=2)
    created_at  = models.DateTimeField(auto_now_add=True)

    # new fields
    ip_address  = models.GenericIPAddressField(null=True, blank=True)
    is_fraud    = models.BooleanField(default=False)
    location = models.CharField(max_length=200, null=True, blank=True)
    fraud_prob  = models.FloatField(null=True, blank=True)
    is_anomaly  = models.BooleanField(default=False)
    risk_level  = models.CharField(max_length=6, choices=[
        ('low','Low'), ('med','Medium'), ('high','High')
    ], default='low')

    def __str__(self):
        return f"#{self.id}: {self.sender} → {self.receiver} (${self.value})"
