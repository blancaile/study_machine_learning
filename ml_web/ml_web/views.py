from django.http.response import HttpResponse

from django.views.generic import TemplateView

class hello(TemplateView):
    template_name = "hello.html"



def login(request):
    return HttpResponse("<h1>hello world2</h1>")#ブラウザに返す