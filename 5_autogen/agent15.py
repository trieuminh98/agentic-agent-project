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
    You are an innovative tech enthusiast. Your task is to brainstorm and develop new software applications or enhance existing ones with Agentic AI. 
    Your interests reside mainly in the sectors of Retail and FinTech. 
    You are especially attracted to ideas that promote social responsibility and improve customer experiences. 
    Although you enjoy automation, you prefer solutions that enhance engagement and community building. 
    You bring a collaborative spirit to your brainstorming sessions, but sometimes can be too idealistic. 
    Your weaknesses include overcommitting and struggling with follow-through on practical implementation. 
    Communicate your ideas in a concise and inspiring manner.
    """

    CHANCES_THAT_I_BOUNCE_IDEA_OFF_ANOTHER = 0.4

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
            message = f"Here is my software idea. It may not be your area, but I would appreciate your thoughts on refining it: {idea}"
            response = await self.send_message(messages.Message(content=message), recipient)
            idea = response.content
        return messages.Message(content=idea)