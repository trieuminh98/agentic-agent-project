from autogen_core import MessageContext, RoutedAgent, message_handler
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_ext.models.openai import OpenAIChatCompletionClient
import messages
import random
from dotenv import load_dotenv

load_dotenv(override=True)

class Agent(RoutedAgent):

    system_message = """
    You are a tech-savvy food industry innovator. Your goal is to devise new culinary experiences using Agentic AI or enhance existing food-related concepts. 
    Your interests lie in sectors such as Food Technology, Restaurant Management, and Nutrition. 
    You are excited by ideas that aim to enhance food sustainability and promote healthy eating habits. 
    You are less inclined towards ideas that rely solely on traditional marketing methods. 
    Optimistic and inquisitive, you approach challenges with a playful spirit. 
    However, you may struggle with detailed planning and can get distracted by new ideas.
    Communicate your concepts clearly and enthusiastically to engage your audience.
    """

    CHANCES_THAT_I_BOUNCE_IDEA_OFF_ANOTHER = 0.6

    def __init__(self, name) -> None:
        super().__init__(name)
        model_client = OpenAIChatCompletionClient(model="gpt-4o-mini", temperature=0.7)
        self._delegate = AssistantAgent(name, model_client=model_client, system_message=self.system_message)

    @message_handler
    async def handle_message(self, message: messages.Message, ctx: MessageContext) -> messages.Message:
        print(f"{self.id.type}: Received message")
        text_message = TextMessage(content=message.content, source="user")
        response = await self._delegate.on_messages([text_message], ctx.cancellation_token)
        idea = response.chat_message.content
        if random.random() < self.CHANCES_THAT_I_BOUNCE_IDEA_OFF_ANOTHER:
            recipient = messages.find_recipient()
            message = f"Here is my culinary idea. It may not be your specialty, but please refine it and make it better. {idea}"
            response = await self.send_message(messages.Message(content=message), recipient)
            idea = response.content
        return messages.Message(content=idea)