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

def add_transaction_view(request):
    form = TransactionForm(request.POST or None)
    if form.is_valid():
        Transaction.objects.create(
            sender=form.cleaned_data['sender'],
            receiver=form.cleaned_data['receiver'],
            value=form.cleaned_data['value']
        )
        return redirect('transactions:landing')
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
