from django.shortcuts import render, redirect
from admin_datta.forms import RegistrationForm, LoginForm, UserPasswordChangeForm, UserPasswordResetForm, UserSetPasswordForm
from django.contrib.auth.views import LoginView, PasswordChangeView, PasswordResetConfirmView, PasswordResetView
from django.views.generic import CreateView
from django.contrib.auth import logout

from django.contrib.auth.decorators import login_required

from .models import *

from .forms import DocumentForm

import os
import shutil
from django.conf import settings

def model_form_upload(request):
    if request.method == 'POST':
        form = DocumentForm(request.POST, request.FILES)
        if form.is_valid():
            empty_media_documents_folder()
            form.save()
            return redirect('conversion')  # Redirect to a new URL
    else:
        form = DocumentForm()
    return render(request, 'pages/conversion.html', {
        'form': form
    })

def index(request):

  context = {
    'segment'  : 'index',
    #'products' : Product.objects.all()
  }
  return render(request, "pages/index.html", context)

def conversion(request):
  context = {
    'segment': 'tables'
  }
  return render(request, "pages/conversion.html", context)

def empty_media_documents_folder():
    folder_path = os.path.join(settings.MEDIA_ROOT, 'documents')
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')