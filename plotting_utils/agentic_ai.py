"""
Modern AI (LLMs, RAG & Agents) helper functions for the agentic AI notebook.

This module contains the data sets, plotting functions and interactive widgets
used by ``05_agentic_ai_rag.ipynb``. As in the other notebooks, the goal is to keep
the notebook focused on the teaching content rather than on plotting code.

Everything in here runs fully offline — no API keys, no downloads.
"""

import textwrap

import plotly.express as px
import plotly.graph_objects as go
from ipywidgets import interact, interact_manual, FloatSlider, IntSlider, Dropdown, Text


# ---------------------------------------------------------------------------
# 1. Data: a tiny text corpus and a small knowledge base
# ---------------------------------------------------------------------------

# A deliberately small corpus. Real language models are trained on trillions of
# tokens — this one has a few thousand. That difference is the whole point:
# it lets us see the mechanism clearly, and it lets us see it fail clearly.
TINY_CORPUS = """
Machine learning is a way of writing programs that learn patterns from data.
Machine learning is used to make predictions about new and unseen examples.
A model is trained on data and then evaluated on data it has never seen.
A model that memorises its training data does not generalise to new data.
Supervised learning needs labelled examples for every task we care about.
Supervised learning works well when labelled data is cheap and plentiful.
Labelled data is expensive to collect and often needs domain experts.
A foundation model is pretrained on enormous amounts of unlabelled text.
A foundation model can then be adapted to many tasks with very little data.
Pretraining gives the model broad general knowledge about language.
Fine tuning adapts a general model to a narrow task or a specific domain.
A medical language model is not built from scratch but fine tuned from a general model.
Scale changes behaviour because more data and more parameters unlock new capabilities.
Capabilities such as translation and summarisation emerge without being trained directly.
A language model predicts the next token given all the previous tokens.
Generating text is simply repeated next token prediction in a loop.
Tokens are word fragments and the model sees language only as tokens.
Tokenisation splits text into tokens before the model ever sees it.
Attention lets the model weight which earlier tokens matter most right now.
Attention is the reason the model can resolve what a pronoun refers to.
The transformer architecture makes attention practical at very large scale.
Every major language model today is built on the transformer architecture.
The model does not know facts and it predicts plausible continuations of text.
A confident sounding answer can still be completely wrong.
Hallucination is a structural property of next token prediction.
Hallucination means the model invents details that sound entirely plausible.
The knowledge cutoff is the date after which the model has seen nothing.
The model cannot know about recent events unless we put them in the prompt.
The context window is the amount of text the model can see at once.
Text outside the context window is simply invisible to the model.
Long documents and long conversations overflow the context window.
The same question phrased differently can produce a different answer.
Language models are good at drafting and summarising and reformatting text.
Language models are good at explaining ideas and brainstorming options.
Language models are weak at precise factual lookup and exact arithmetic.
Retrieval augmented generation grounds the answer in documents we control.
Retrieval augmented generation retrieves relevant documents at query time.
The retrieved documents are injected into the prompt before generation.
The model then answers using the retrieved text instead of its memory.
Embeddings represent documents as vectors in a high dimensional space.
Documents with similar meaning have similar vectors in embedding space.
Retrieval finds the nearest neighbours of the query vector.
An open book exam is easier than a closed book exam for the same reason.
Grounding the answer in sources makes the answer checkable by a human.
A citation lets the reader verify the claim against the original document.
An agent uses a language model to plan and act and observe and iterate.
An agent decides what to do next based on what happened so far.
Tool use lets the agent call a calculator or a search engine or an api.
Tool use extends the model far beyond generating text alone.
A tool returns an observation and the observation goes back into the context.
Short term memory is the conversation held inside the context window.
Long term memory is stored outside the model and retrieved when needed.
Working memory is the scratch pad the agent writes while it reasons.
The react pattern alternates between reasoning steps and action steps.
Chain of thought decomposes a hard problem into smaller steps before answering.
A research agent searches for sources and extracts passages and cites them.
A coding agent writes code and runs it and reads the error and fixes the bug.
A multi agent system gives different roles to different agents.
One agent drafts the text and one agent checks the facts and one agent edits.
An agent that takes actions in the world can cause real consequences.
An agent loop needs a step limit so that it cannot run forever.
Irreversible actions should require a human to approve them first.
Reliability and controllability and accountability are open problems.
Evaluation of agents is much harder than evaluation of a single answer.
A system that acts is not the same as a system that answers questions.
"""


# A small, self-contained knowledge base for the retrieval examples.
# Each entry is a short factual note with a source label we can cite.
KNOWLEDGE_BASE = [
    {
        "id": "kb-01",
        "source": "Course handbook, Session 1",
        "text": (
            "Supervised learning requires labelled examples. For every input in the "
            "training set we also need the correct answer. Collecting these labels is "
            "usually the most expensive part of a machine learning project."
        ),
    },
    {
        "id": "kb-02",
        "source": "Course handbook, Session 2",
        "text": (
            "A decision tree splits the data with a sequence of yes or no questions. "
            "Random forests combine many trees and average their votes, which reduces "
            "overfitting and usually improves accuracy."
        ),
    },
    {
        "id": "kb-03",
        "source": "Course handbook, Session 3",
        "text": (
            "A neural network stacks layers of artificial neurons. Each neuron computes "
            "a weighted sum of its inputs and applies a non-linear activation function. "
            "Depth lets the network build abstract features from raw inputs."
        ),
    },
    {
        "id": "kb-04",
        "source": "Session 4 notes: foundation models",
        "text": (
            "A foundation model is pretrained on very large amounts of unlabelled data "
            "and can then be adapted to many downstream tasks. Adaptation needs far less "
            "labelled data than training a task specific model from scratch."
        ),
    },
    {
        "id": "kb-05",
        "source": "Session 4 notes: how LLMs work",
        "text": (
            "A large language model is trained to predict the next token given the "
            "preceding context. Generating a sentence is nothing but repeated next token "
            "prediction, feeding each predicted token back in as input."
        ),
    },
    {
        "id": "kb-06",
        "source": "Session 4 notes: how LLMs work",
        "text": (
            "Attention is the mechanism that lets a model weigh which earlier words in the "
            "context are relevant to the word it is predicting right now. Attention is how "
            "the model resolves what a pronoun refers to. The transformer is the "
            "architecture that makes attention work efficiently at very large scale."
        ),
    },
    {
        "id": "kb-07",
        "source": "Session 4 notes: failure modes",
        "text": (
            "Hallucination means that a language model makes things up. When it does not "
            "know something it does not stop: it will make up a fact, a name, a number or "
            "a citation that sounds entirely plausible and is simply wrong. This follows "
            "directly from the training objective, because the model optimises for "
            "plausible continuations of text and never for truth."
        ),
    },
    {
        "id": "kb-08",
        "source": "Session 4 notes: failure modes",
        "text": (
            "The knowledge cutoff is the point in time after which the model has seen no "
            "training data. Anything that happened later must be supplied in the prompt, "
            "otherwise the model will either refuse or invent an answer."
        ),
    },
    {
        "id": "kb-09",
        "source": "Session 4 notes: failure modes",
        "text": (
            "The context window is the maximum amount of text a model can attend to in a "
            "single request. Content that does not fit is invisible to the model, which "
            "is why very long documents have to be split into chunks."
        ),
    },
    {
        "id": "kb-10",
        "source": "Session 4 notes: RAG",
        "text": (
            "Retrieval augmented generation, or RAG, retrieves relevant documents from an "
            "external knowledge base at query time and injects them into the prompt. The "
            "model then answers from the retrieved text rather than from memory."
        ),
    },
    {
        "id": "kb-11",
        "source": "Session 4 notes: RAG",
        "text": (
            "In a RAG pipeline the query and the documents are turned into embedding "
            "vectors. Retrieval is a nearest neighbour search: the documents whose vectors "
            "are closest to the query vector are assumed to be the most relevant."
        ),
    },
    {
        "id": "kb-12",
        "source": "Session 4 notes: agents",
        "text": (
            "An agent is a system that uses a language model in a loop: it plans a step, "
            "takes an action such as calling a tool, observes the result, and then decides "
            "what to do next. A single answer becomes a sequence of decisions."
        ),
    },
    {
        "id": "kb-13",
        "source": "Session 4 notes: agents",
        "text": (
            "The ReAct pattern interleaves reasoning and acting: the agent writes a thought, "
            "chooses an action, reads the observation, and repeats. Chain of thought is the "
            "related idea of decomposing a problem into steps before answering."
        ),
    },
    {
        "id": "kb-14",
        "source": "Session 4 notes: agents",
        "text": (
            "Agent memory comes in three flavours. Short term memory is the conversation in "
            "the context window. Long term memory is stored in an external database and "
            "retrieved on demand. Working memory is the scratch pad of the current task."
        ),
    },
    {
        "id": "kb-15",
        "source": "Session 4 notes: risks",
        "text": (
            "Agents that take actions raise new safety questions. Practical guardrails "
            "include a hard limit on the number of steps, a whitelist of allowed tools, "
            "logging of every action, and human approval before irreversible actions."
        ),
    },
    {
        "id": "kb-16",
        "source": "Session 4 notes: costs",
        "text": (
            "Running a large language model costs money. Commercial providers charge a "
            "price per token, for the input tokens in the prompt as well as for the output "
            "tokens in the response. A long retrieved context therefore makes every single "
            "request more expensive, so cost and answer quality must be traded off."
        ),
    },
]


# ---------------------------------------------------------------------------
# 2. Tokenisation & next-token prediction
# ---------------------------------------------------------------------------

def plot_token_counts(examples):
    """Compare words vs. tokens for a few example strings.

    ``examples`` is a list of ``(label, n_words, n_tokens)`` tuples.
    """
    labels = [textwrap.shorten(e[0], 30, placeholder="…") for e in examples]
    fig = go.Figure()
    fig.add_bar(x=labels, y=[e[1] for e in examples], name="Words", marker_color="#7fb3d5")
    fig.add_bar(x=labels, y=[e[2] for e in examples], name="Tokens", marker_color="#e59866")
    fig.update_layout(
        title="🔤 Words vs. tokens — the model counts tokens, not words",
        xaxis_title="Example text",
        yaxis_title="Count",
        barmode="group",
    )
    fig.show()


def create_interactive_tokenizer(tokenize_fn, price_per_million=5.0):
    """Type any text and see it as the model sees it — tokens, and what they cost."""
    @interact_manual(
        text=Text(value="Bridging AI & Society — a hands-on summer school.",
                  description="Your text:", layout={"width": "80%"}),
        requests_per_day=IntSlider(value=1000, min=0, max=100000, step=1000,
                                   description="Req./day"),
    )
    def show(text, requests_per_day):
        tokens = tokenize_fn(text)
        if not tokens:
            print("Type something above, then press Run Interact.")
            return
        print(f"📝 {len(text)} characters   →   {len(text.split())} words   "
              f"→   {len(tokens)} tokens")
        print(f"   {len(text) / len(tokens):.1f} characters per token\n")
        print("🔤 " + " | ".join(tokens) + "\n")
        cost = len(tokens) / 1_000_000 * price_per_million
        print(f"💸 At ${price_per_million:.2f} per million input tokens:")
        print(f"   one request:        ${cost:.6f}")
        print(f"   {requests_per_day:,} requests/day: ${cost * requests_per_day:9.2f} per day"
              f"   (${cost * requests_per_day * 365:,.0f} per year)")
        print("   ...and that is the prompt alone. The answer is billed too, "
              "usually at 3–5× the rate.")


def plot_next_token_distribution(context, distribution, top_k=10):
    """Bar chart of the model's probability distribution over the next token."""
    items = sorted(distribution.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
    if not items:
        print("The model has never seen this context — no prediction possible.")
        return

    tokens = [t for t, _ in items]
    probs = [p for _, p in items]

    fig = px.bar(
        x=tokens, y=probs,
        title=f'🎲 What comes after "…{context}"?',
        labels={"x": "Candidate next token", "y": "Probability"},
        color=probs, color_continuous_scale="Blues",
    )
    fig.update_layout(coloraxis_showscale=False, yaxis_tickformat=".0%")
    fig.show()


def create_interactive_text_generator(build_model, prompts):
    """Slider-driven text generation: context length, temperature, seed.

    ``build_model(order)`` must return an object with a
    ``generate(prompt, n_tokens, temperature, seed)`` method.
    """
    cache = {}

    @interact(
        prompt=Dropdown(options=prompts, description="Prompt:"),
        order=IntSlider(value=2, min=1, max=4, step=1, description="Context (n)"),
        temperature=FloatSlider(value=1.0, min=0.1, max=2.0, step=0.1, description="Temperature"),
        n_tokens=IntSlider(value=40, min=10, max=80, step=5, description="Length"),
        seed=IntSlider(value=0, min=0, max=20, step=1, description="Seed"),
    )
    def generate(prompt, order, temperature, n_tokens, seed):
        if order not in cache:
            cache[order] = build_model(order)
        text = cache[order].generate(prompt, n_tokens=n_tokens,
                                     temperature=temperature, seed=seed)
        print(f"context = last {order} token(s)   |   temperature = {temperature}\n")
        print(textwrap.fill(text, width=88))


def plot_attention_illustration():
    """Illustrate attention with the classic 'trophy / suitcase' ambiguity.

    NOTE: these weights are hand-crafted for teaching. They show what attention
    *does*, not what any particular model actually computed.
    """
    words = ["The", "trophy", "didn't", "fit", "in", "the", "suitcase",
             "because", "it", "was", "too", "___"]
    big = [0.01, 0.55, 0.02, 0.05, 0.01, 0.01, 0.18, 0.03, 0.10, 0.02, 0.02, 0.00]
    small = [0.01, 0.17, 0.02, 0.05, 0.01, 0.01, 0.56, 0.03, 0.10, 0.02, 0.02, 0.00]

    fig = go.Figure()
    fig.add_bar(x=words, y=big, name='… too BIG   → "it" = the trophy',
                marker_color="#5499c7")
    fig.add_bar(x=words, y=small, name='… too SMALL → "it" = the suitcase',
                marker_color="#e59866")
    fig.update_layout(
        title='🔍 Attention: which word does "it" refer to?',
        xaxis_title="Words in the sentence",
        yaxis_title='Attention weight given to each word when reading "it"',
        barmode="group",
    )
    fig.show()


# ---------------------------------------------------------------------------
# 3. Limitations: the context window
# ---------------------------------------------------------------------------

def create_interactive_context_window(document, needle, answer_fn):
    """Show what the model can still 'see' as the context window shrinks.

    ``document`` is a list of sentences, ``needle`` is the index of the sentence
    holding the key fact, and ``answer_fn(visible_sentences)`` returns the answer
    the model would give with that much context.
    """
    total = len(document)

    @interact(window=IntSlider(value=total, min=1, max=total, step=1,
                               description="Window"))
    def show(window):
        visible = document[-window:]
        first_visible = total - window
        print(f"📏 Context window: {window} of {total} sentences\n")
        for i, sentence in enumerate(document):
            if i < first_visible:
                print(f"   ✂️  [cut]  {sentence}")
            else:
                marker = "🔑" if i == needle else "  "
                print(f"   👁️  {marker}     {sentence}")
        print("\n" + "-" * 80)
        print(f"❓ Question: Which room is the workshop in?")
        print(f"🤖 Answer:   {answer_fn(visible)}")


# ---------------------------------------------------------------------------
# 4. Retrieval-augmented generation
# ---------------------------------------------------------------------------

def plot_retrieval_scores(query, results, threshold=None):
    """Bar chart of similarity scores for the retrieved documents."""
    ids = [r["id"] for r in results]
    scores = [r["score"] for r in results]

    fig = px.bar(
        x=ids, y=scores,
        title=f'🔎 Similarity to the query: "{query}"',
        labels={"x": "Document", "y": "Similarity (0 = unrelated, 1 = identical)"},
        color=scores, color_continuous_scale="Greens", range_color=(0, 1),
    )
    if threshold is not None:
        fig.add_hline(y=threshold, line_dash="dash", line_color="red",
                      annotation_text="relevance threshold")
    fig.update_layout(coloraxis_showscale=False, yaxis_range=[0, 1])
    fig.show()


def plot_document_space(doc_coords, doc_ids, doc_sources, query_coords=None, query_text="",
                        doc_groups=None, doc_hover=None):
    """2-D projection of the document vectors, plus the query vector.

    ``doc_groups`` optionally assigns each document to a category, so that
    topically related documents get the same colour.
    """
    groups = doc_groups if doc_groups is not None else ["Documents"] * len(doc_ids)
    hover = doc_hover if doc_hover is not None else doc_sources
    palette = ["#5499c7", "#e59866", "#48c9b0", "#af7ac5", "#f7dc6f", "#ec7063"]

    fig = go.Figure()
    for n, group in enumerate(dict.fromkeys(groups)):
        colour = palette[n % len(palette)]          # cycle, so no group is ever dropped
        picks = [i for i, g in enumerate(groups) if g == group]
        fig.add_trace(go.Scatter(
            x=[doc_coords[i, 0] for i in picks],
            y=[doc_coords[i, 1] for i in picks],
            mode="markers+text",
            text=[doc_ids[i] for i in picks],
            textposition="top center",
            marker=dict(size=12, color=colour),
            hovertext=[hover[i] for i in picks],
            hoverinfo="text",
            name=group,
        ))
    if query_coords is not None:
        fig.add_trace(go.Scatter(
            x=[query_coords[0]], y=[query_coords[1]],
            mode="markers+text",
            text=["QUERY"],
            textposition="bottom center",
            marker=dict(size=18, color="#c0392b", symbol="star"),
            hovertext=[query_text],
            name="Query",
        ))
    fig.update_layout(
        title="🗺️ The knowledge base in embedding space (2-D projection)",
        xaxis_title="Component 1", yaxis_title="Component 2",
    )
    fig.show()


def create_interactive_rag_explorer(rag_answer_fn, example_queries):
    """Type a question, see what gets retrieved and what the grounded answer is."""
    print("💡 Try one of these, or write your own:")
    for q in example_queries:
        print(f"   • {q}")
    print()

    @interact_manual(
        question=Text(value=example_queries[0], description="Ask:",
                      layout={"width": "80%"}),
        top_k=IntSlider(value=3, min=1, max=5, step=1, description="Top-k"),
    )
    def ask(question, top_k):
        print(rag_answer_fn(question, top_k=top_k))


# ---------------------------------------------------------------------------
# 5. Agents
# ---------------------------------------------------------------------------

def print_agent_trace(trace):
    """Pretty-print a ReAct trace: thought → action → observation → …"""
    print("=" * 84)
    print(f"🎯 GOAL: {trace['goal']}")
    print("=" * 84)
    for step in trace["steps"]:
        print(f"\n── Step {step['n']} " + "─" * 68)
        print(f"  🤔 Thought:     {step['thought']}")
        print(f"  🔧 Action:      {step['action']}({step['action_input']!r})")
        observation = textwrap.fill(
            str(step["observation"]), width=80,
            initial_indent="  👀 Observation: ", subsequent_indent="                  ",
        )
        print(observation)
    icon = "✅" if trace["status"].startswith("finished") else "🛑"
    print("\n" + "=" * 84)
    print(f"{icon} RESULT: {trace['answer']}")
    print(f"   ({len(trace['steps'])} step(s), {trace['status']})")
    print("=" * 84)


def plot_agent_trace(trace):
    """Visualise the plan → act → observe loop as a step timeline."""
    steps = trace["steps"]
    if not steps:
        print("No steps to plot.")
        return

    labels = [f"Step {s['n']}<br>{s['action']}" for s in steps]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(1, len(steps) + 1)),
        y=[1] * len(steps),
        mode="markers+lines+text",
        text=labels,
        textposition="top center",
        marker=dict(size=26, color="#48c9b0", line=dict(width=2, color="#148f77")),
        line=dict(color="#148f77", width=2),
        hovertext=[f"Thought: {s['thought']}<br>Observation: {s['observation']}"
                   for s in steps],
        hoverinfo="text",
        showlegend=False,
    ))
    fig.update_layout(
        title=f"🔁 The agent loop — {textwrap.shorten(trace['goal'], 70)}",
        xaxis_title="Iteration",
        yaxis=dict(visible=False, range=[0.5, 1.6]),
        height=320,
    )
    fig.show()


def plot_tool_usage_summary(traces):
    """How often each tool was called across a set of agent runs."""
    counts = {}
    for trace in traces:
        for step in trace["steps"]:
            counts[step["action"]] = counts.get(step["action"], 0) + 1
    if not counts:
        print("No tool calls recorded.")
        return

    tools = list(counts.keys())
    fig = px.bar(
        x=tools, y=[counts[t] for t in tools],
        title="🔧 Which tools did the agent reach for?",
        labels={"x": "Tool", "y": "Number of calls"},
        color=tools,
    )
    fig.update_layout(showlegend=False)
    fig.show()
