from django.db import models

class Transaction(models.Model):
    sender      = models.CharField(max_length=100)
    receiver    = models.CharField(max_length=100)
    value       = models.DecimalField(max_digits=10, decimal_places=2)
    created_at  = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"#{self.id}: {self.sender} → {self.receiver} (${self.value})"
