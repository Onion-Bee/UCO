from django import forms

# forms.py
from django import forms

class TransactionForm(forms.Form):
    receiver = forms.CharField(max_length=100)
    value    = forms.DecimalField(max_digits=10, decimal_places=2, min_value=0)


class RandomTransactionsForm(forms.Form):
    count     = forms.IntegerField(min_value=1, label="Number of transactions (n)")
    max_value = forms.DecimalField(max_digits=10, decimal_places=2, min_value=0, label="Max value per txn")
