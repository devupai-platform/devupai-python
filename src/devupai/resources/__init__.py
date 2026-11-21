"""The API resources of :class:`devupai.DevupAI` and :class:`devupai.AsyncDevupAI`.

Each resource with API methods has a ``with_response`` twin: the same methods, returning the whole
:class:`devupai.APIResponse` instead of its data.
"""

from devupai.resources.audio import (
    AsyncAudio,
    AsyncSpeech,
    AsyncSpeechWithResponse,
    AsyncTranscriptions,
    AsyncTranscriptionsWithResponse,
    Audio,
    Speech,
    SpeechWithResponse,
    Transcriptions,
    TranscriptionsWithResponse,
)
from devupai.resources.balance import AsyncBalance, AsyncBalanceWithResponse, Balance, BalanceWithResponse
from devupai.resources.chat import (
    AsyncChat,
    AsyncCompletions,
    AsyncCompletionsWithResponse,
    Chat,
    Completions,
    CompletionsWithResponse,
)
from devupai.resources.embeddings import (
    AsyncEmbeddings,
    AsyncEmbeddingsWithResponse,
    Embeddings,
    EmbeddingsWithResponse,
)
from devupai.resources.health import AsyncHealth, AsyncHealthWithResponse, Health, HealthWithResponse
from devupai.resources.images import AsyncImages, AsyncImagesWithResponse, Images, ImagesWithResponse
from devupai.resources.inference import (
    AsyncNativeInference,
    AsyncNativeInferenceWithResponse,
    NativeInference,
    NativeInferenceWithResponse,
)
from devupai.resources.me import AsyncMe, AsyncMeWithResponse, Me, MeWithResponse
from devupai.resources.models import AsyncModels, AsyncModelsWithResponse, Models, ModelsWithResponse
from devupai.resources.rerank import AsyncRerank, AsyncRerankWithResponse, Rerank, RerankWithResponse
from devupai.resources.responses import (
    AsyncResponses,
    AsyncResponsesWithResponse,
    Responses,
    ResponsesWithResponse,
)
from devupai.resources.video import (
    AsyncVideo,
    AsyncVideoEdits,
    AsyncVideoEditsWithResponse,
    AsyncVideoGenerations,
    AsyncVideoGenerationsWithResponse,
    Video,
    VideoEdits,
    VideoEditsWithResponse,
    VideoGenerations,
    VideoGenerationsWithResponse,
)

__all__ = [
    # chat
    "Chat",
    "AsyncChat",
    "Completions",
    "AsyncCompletions",
    "CompletionsWithResponse",
    "AsyncCompletionsWithResponse",
    # responses
    "Responses",
    "AsyncResponses",
    "ResponsesWithResponse",
    "AsyncResponsesWithResponse",
    # models
    "Models",
    "AsyncModels",
    "ModelsWithResponse",
    "AsyncModelsWithResponse",
    # images
    "Images",
    "AsyncImages",
    "ImagesWithResponse",
    "AsyncImagesWithResponse",
    # embeddings
    "Embeddings",
    "AsyncEmbeddings",
    "EmbeddingsWithResponse",
    "AsyncEmbeddingsWithResponse",
    # rerank
    "Rerank",
    "AsyncRerank",
    "RerankWithResponse",
    "AsyncRerankWithResponse",
    # audio
    "Audio",
    "AsyncAudio",
    "Speech",
    "AsyncSpeech",
    "SpeechWithResponse",
    "AsyncSpeechWithResponse",
    "Transcriptions",
    "AsyncTranscriptions",
    "TranscriptionsWithResponse",
    "AsyncTranscriptionsWithResponse",
    # video
    "Video",
    "AsyncVideo",
    "VideoGenerations",
    "AsyncVideoGenerations",
    "VideoGenerationsWithResponse",
    "AsyncVideoGenerationsWithResponse",
    "VideoEdits",
    "AsyncVideoEdits",
    "VideoEditsWithResponse",
    "AsyncVideoEditsWithResponse",
    # inference
    "NativeInference",
    "AsyncNativeInference",
    "NativeInferenceWithResponse",
    "AsyncNativeInferenceWithResponse",
    # balance, me, health
    "Balance",
    "AsyncBalance",
    "BalanceWithResponse",
    "AsyncBalanceWithResponse",
    "Me",
    "AsyncMe",
    "MeWithResponse",
    "AsyncMeWithResponse",
    "Health",
    "AsyncHealth",
    "HealthWithResponse",
    "AsyncHealthWithResponse",
]
