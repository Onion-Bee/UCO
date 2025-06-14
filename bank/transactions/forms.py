from django import forms

class TransactionForm(forms.Form):
    sender   = forms.CharField(max_length=100)
    receiver = forms.CharField(max_length=100)
    value    = forms.DecimalField(max_digits=10, decimal_places=2, min_value=0)

class RandomTransactionsForm(forms.Form):
    count     = forms.IntegerField(min_value=1, label="Number of transactions (n)")
    max_value = forms.DecimalField(max_digits=10, decimal_places=2, min_value=0, label="Max value per txn")
