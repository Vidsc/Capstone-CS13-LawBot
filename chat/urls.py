from django.urls import path
from . import views

urlpatterns = [
    # 主页：右侧聊天 + 左侧会话列表
    path("", views.chat_page, name="chat_page"),

    # 会话管理
    path("chat/new/", views.new_chat, name="new_chat"),
    path("chat/<str:chat_id>/", views.load_chat, name="load_chat"),

    # 消息交互
    path("send", views.send_message, name="send_message"),
    path("reset", views.reset_chat, name="reset_chat"),
]
