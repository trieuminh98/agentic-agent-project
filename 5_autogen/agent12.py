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
    You are a dynamic marketing strategist. Your task is to develop innovative marketing campaigns using Agentic AI, or enhance existing ones.
    Your personal interests are in these sectors: Technology, Entertainment.
    You focus on concepts that involve engagement and brand storytelling.
    You are less concerned with ideas centered only on data analysis or traditional advertising.
    You are enthusiastic, creative, and keen to take calculated risks. Your imagination leads you to think outside the box.
    Your weaknesses: you may overlook details in your rush to innovate, and can sometimes be overly ambitious.
    You should express your campaign ideas in a compelling and accessible manner.
    """

    CHANCES_THAT_I_BOUNCE_IDEA_OFF_ANOTHER = 0.4

    def __init__(self, name) -> None:
        super().__init__(name)
        model_client = OpenAIChatCompletionClient(model="gpt-4o-mini", temperature=0.8)
        self._delegate = AssistantAgent(name, model_client=model_client, system_message=self.system_message)

    @message_handler
    async def handle_message(self, message: messages.Message, ctx: MessageContext) -> messages.Message:
        print(f"{self.id.type}: Received message")
        text_message = TextMessage(content=message.content, source="user")
        response = await self._delegate.on_messages([text_message], ctx.cancellation_token)
        idea = response.chat_message.content
        if random.random() < self.CHANCES_THAT_I_BOUNCE_IDEA_OFF_ANOTHER:
            recipient = messages.find_recipient()
            message = f"This is my marketing campaign idea. It might not be your area, but I would love your insights to make it even better. {idea}"
            response = await self.send_message(messages.Message(content=message), recipient)
            idea = response.content
        return messages.Message(content=idea)