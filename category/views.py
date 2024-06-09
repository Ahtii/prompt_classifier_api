from rest_framework import views
from rest_framework.response import Response

from category import utils
from category import validators


class CategoryAPIView(views.APIView):

    def post(self, request, *args, **kwargs):
        data = request.data
        prompt = validators.validate_prompt(data.get('prompt'))
        chat_open_ai = utils.ChatOpenAIUtils(prompt)
        response = chat_open_ai.get_response()
        return Response(response)
