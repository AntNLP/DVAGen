import json
import traceback

from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.http import require_POST

from generation.utils import generate_util


# Create your views here.
def index(request):
    return render(request, "generation/index.html")


@require_POST
def generate(request):
    try:
        data = json.loads(request.body)
        prefix = data.get("prefix")
        phrases = data.get("phrases", [])
        results = generate_util(prefix, phrases)
        print(
            "[Backend]",
            "Prefix:",
            prefix,
            "Phrases:",
            ", ".join(phrases),
            "Result:",
            "".join([item["chosenToken"] for item in results]),
            sep="\n",
        )

        # 返回 JSON 响应给前端
        return JsonResponse(
            {
                "status": "success",
                "results": results,
                "message": "Generated successfully!",
            }
        )
    except json.JSONDecodeError:
        print(traceback.format_exc())
        return JsonResponse(
            {"status": "error", "message": "Error JSON data"},
            status=400,
        )
    except Exception as e:
        print(traceback.format_exc())
        return JsonResponse(
            {"status": "error", "message": str(e)},
            status=500,
        )
