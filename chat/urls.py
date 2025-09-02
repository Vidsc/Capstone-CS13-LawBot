from django.urls import path
from . import views

urlpatterns = [
    path("", views.chat_page, name="chat_page"),
    path("send", views.send_message, name="send_message"),
    path("reset", views.reset_chat, name="reset_chat"),
]
