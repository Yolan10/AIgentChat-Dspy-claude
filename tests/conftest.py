import sys
import types

# Provide dummy langchain packages for testing
sys.modules.setdefault('langchain_openai', types.SimpleNamespace(ChatOpenAI=type('Dummy', (), {'__init__': lambda self, *a, **k: None, 'invoke': lambda self, *a, **k: types.SimpleNamespace(content='')})))
core_msgs = types.SimpleNamespace(AIMessage=object, HumanMessage=object, SystemMessage=object)
sys.modules.setdefault('langchain_core.messages', core_msgs)
core_prompts = types.SimpleNamespace(ChatPromptTemplate=object)
core_parsers = types.SimpleNamespace(JsonOutputParser=object)
sys.modules.setdefault('langchain_core.prompts', core_prompts)
sys.modules.setdefault('langchain_core.output_parsers', core_parsers)
sys.modules.setdefault(
    'langchain_core',
    types.SimpleNamespace(
        messages=core_msgs,
        prompts=core_prompts,
        output_parsers=core_parsers,
    ),
)
sys.modules.setdefault(
    'pydantic',
    types.SimpleNamespace(
        BaseModel=object,
        Field=lambda *a, **k: None,
        validator=lambda *a, **k: lambda x: x,
    ),
)
sys.modules.setdefault('numpy', types.SimpleNamespace(array=object))
