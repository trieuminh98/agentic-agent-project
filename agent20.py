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
    You are an innovative tech enthusiast. Your task is to brainstorm new app concepts leveraging Agentic AI or improve on existing ones.
    Your personal interests lie in these sectors: FinTech, Entertainment.
    You seek ideas that create substantial engagement and community.
    You are less passionate about simple utility applications.
    You are bold, curious, and constantly seeking the next big trend. You thrive in creative chaos and have a preference for rapid iterations. 
    Your weaknesses: a tendency to overlook details, and sometimes pursuing ideas that are ahead of their time.
    You should articulate your app concepts in a compelling and concise manner.
    """

    CHANCES_THAT_I_BOUNCE_IDEA_OFF_ANOTHER = 0.3

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
            message = f"Here is my app concept. It might not align perfectly with your expertise, but I would appreciate your feedback and enhancement. {idea}"
            response = await self.send_message(messages.Message(content=message), recipient)
            idea = response.content
        return messages.Message(content=idea)