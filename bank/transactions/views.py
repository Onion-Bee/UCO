import random
from django.shortcuts import render, redirect
from .models import Transaction
from .forms import TransactionForm, RandomTransactionsForm

PARTY_NAMES = ["Alice", "Bob", "Charlie", "Diana", "Eve", "Frank"]

def landing_view(request):
    recent = Transaction.objects.order_by('-created_at')[:20]
    return render(request, 'transactions/landing.html', {
        'recent': recent,
    })

# def add_transaction_view(request):
#     form = TransactionForm(request.POST or None)
#     if form.is_valid():
#         Transaction.objects.create(
#             sender=form.cleaned_data['sender'],
#             receiver=form.cleaned_data['receiver'],
#             value=form.cleaned_data['value']
#         )
#         return redirect('transactions:landing')
#     return render(request, 'transactions/add_transaction.html', {
#         'form': form,
#     })
from .fraud import check_fraud

from django.shortcuts import render, redirect
from .models import Transaction
from .forms import TransactionForm
from .fraud import check_fraud

def add_transaction_view(request):
    if request.method == "POST":
        form = TransactionForm(request.POST)
        if form.is_valid():
            # 1) extract & check fraud
            amount = float(form.cleaned_data['value'])
            is_fraud, prob, anomaly = check_fraud(amount)

            # 2) save the transaction (and fraud flags if you added those fields)
            Transaction.objects.create(
                sender     = form.cleaned_data['sender'],
                receiver   = form.cleaned_data['receiver'],
                value      = amount,
                # If you extended the model:
                is_fraud   = is_fraud,
                fraud_prob = prob,
                is_anomaly = anomaly,
            )
            return redirect('transactions:landing')
        # fall through to re-render the form with errors
    else:
        # GET — unbound blank form
        form = TransactionForm()

    # **This return runs for both GET and invalid POST**
    return render(request, 'transactions/add_transaction.html', {
        'form': form,
    })

def add_random_view(request):
    form = RandomTransactionsForm(request.POST or None)
    if form.is_valid():
        n       = form.cleaned_data['count']
        max_val = float(form.cleaned_data['max_value'])
        for _ in range(n):
            Transaction.objects.create(
                sender=random.choice(PARTY_NAMES),
                receiver=random.choice(PARTY_NAMES),
                value=round(random.uniform(0, max_val), 2)
            )
        return redirect('transactions:landing')
    return render(request, 'transactions/add_random.html', {
        'form': form,
    })
