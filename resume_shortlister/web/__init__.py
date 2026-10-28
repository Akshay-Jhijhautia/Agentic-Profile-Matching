"""The local web app: ``resume-shortlister web`` serves an API and the browser UI on 127.0.0.1.

- ``models``: the API's response shapes (published as OpenAPI for the UI)
- ``workspace``: the open chats, one ``Conversation`` and resume pool per session
- ``app``: the FastAPI app, its local-only guard and ``serve``
"""
