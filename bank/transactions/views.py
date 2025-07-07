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

from django.shortcuts import render, redirect
from .models import Transaction
from .forms import TransactionForm, RandomTransactionsForm
from .fraud import check_fraud


import requests
from django.shortcuts import render, redirect
from .models import Transaction
from .forms import TransactionForm, RandomTransactionsForm
from .fraud import check_fraud

def add_transaction_view(request):
    if request.method == "POST":
        form = TransactionForm(request.POST)
        if form.is_valid():
            amount = float(form.cleaned_data['value'])
            is_fraud, prob, anomaly = check_fraud(amount)

            # 3‑tier risk
            if prob >= 0.7:
                risk = 'high'
            elif prob >= 0.3:
                risk = 'med'
            else:
                risk = 'low'

            # get client IP
            ip = request.META.get('HTTP_X_FORWARDED_FOR')
            if ip:
                ip = ip.split(',')[0].strip()
            else:
                ip = request.META.get('REMOTE_ADDR')

            # call ipapi.co
            try:
                resp = requests.get(f"https://ipapi.co/{ip}/json/", timeout=2)
                data = resp.json()
                city    = data.get("city", "")
                region  = data.get("region", "")
                country = data.get("country_name", "")
                location = ", ".join(part for part in (city, region, country) if part)
            except Exception:
                location = "Unknown"

            # save
            Transaction.objects.create(
                receiver   = form.cleaned_data['receiver'],
                value      = amount,
                ip_address = ip,
                location   = location,
                is_fraud   = is_fraud,
                fraud_prob = round(prob, 3),
                is_anomaly = anomaly,
                risk_level = risk,
            )
            return redirect('transactions:landing')
    else:
        form = TransactionForm()

    return render(request, 'transactions/add_transaction.html', {'form': form})


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
