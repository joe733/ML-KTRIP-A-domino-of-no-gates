# ML-KTRIP-A-domino-of-no-gates

# A Domino of No-Gates: How I Somehow Ended Up Building an AI Inference Engine

*The story of ML-KTRIP, a rather peculiar mathematical idea, and what happens when you keep asking a question.*

I genuinely don't quite know where to begin.

Actually, that's not true. I know exactly where to begin.

It started with a question.

Not a particularly sophisticated question, either. More a little thought experiment about mathematics, memory, and whether computers really need to go looking for the same answers over and over again.

Imagine you have a mathematical problem.

You ask someone if they know the answer.

They don't.

But perhaps their neighbour does.

So they ask their neighbour, who asks another neighbour, who eventually finds someone who can resolve the problem.

The answer travels back through the chain.

Simple enough.

But then I thought: **why would we ever need to make that exact journey again?**

Why not remember the mathematical path that worked? And, importantly, why not remember the places that couldn't help?

Not a blanket declaration that those places are useless. Just that, for this particular mathematical question, within these particular circumstances, they couldn't resolve it.

A little domino of yeses and noes.

And then the next question comes along.

Some of the mathematics is familiar. Some of the paths are already known. Some aren't.

So we start where we left off.

And this is where things began to get rather interesting.

## The original thought: no-gates

I called them *no-gates*.

I wanted a mathematical system where a node could essentially say:

"No, I can't answer that. And neither can the relevant neighbours I've already checked."

Or:

"Yes. I know where that answer comes from."

Or, just as importantly:

"I don't know yet."

That last one matters.

A system shouldn't confidently say no when it simply hasn't finished looking.

The distinction between **NO** and **UNKNOWN** might sound insignificant, but it's fundamental to making remembered routes trustworthy.

Over time, the system builds a record of successful routes, failed routes, the mathematical conditions under which they were valid, and when they need reconsidering.

It isn't learning new facts about the universe by magic.

It's learning how its own mathematics can be resolved.

And I wondered whether we could apply that idea to machine learning.

Not to replace the mathematics of a neural network.

To execute it differently.

## Introducing PyThreads

This led me to something I called a *PyThread*.

The idea was to give a computational process an extraordinarily small memory of its mathematical journey.

Originally, I set a target of just 50 KiB.

Fifty kilobytes.

Not enough to hold a language model. Not enough to hold its enormous weight matrices. Not even necessarily enough to hold all its intermediate numerical values.

And that's not what I wanted it to do.

I wanted a PyThread to hold the mathematical continuation.

What question are we resolving? Where are we in the calculation? What dependencies have been satisfied? Which routes worked? Which couldn't? Where is the larger state stored? What needs to happen next?

A PyThread is rather like a tiny notebook travelling through an enormous mathematical landscape.

It doesn't carry the landscape.

It carries the directions, the current position, the applicable rules, and references to the work already completed.

Larger numerical states, such as attention caches, can remain in external storage.

And because the PyThread is serialisable, the process can stop, preserve its position and resume without having to rediscover everything from the beginning.

That was the idea.

But then I asked myself a considerably more troublesome question.

**What if we could use this to run an enormous language model with very little RAM?**

## The thing about model weights

When we talk about a 36-billion-parameter model, or a model containing hundreds of billions or even trillions of parameters, we're talking about an enormous collection of learned numerical values.

The weights.

That is where the learned parameters of the model live.

And ordinarily, efficiently running a large model involves keeping substantial portions of those weights in RAM or GPU memory.

Naturally, this creates a practical problem.

If your computer doesn't have enough memory, you're limited in which models you can comfortably run.

But I kept coming back to something.

**Why are we treating the size of the stored model as though it ought to determine the size of the active computational workspace?**

The weights are static during ordinary inference.

They don't need to change every time I ask the model a question.

They're stored intelligence, if you like.

The thing that changes is the computation performed with them.

So I started thinking about the model differently.

Imagine the complete model sitting on an SSD.

All of it.

Not compressed into some lesser version. Not distilled into a smaller model. Not missing half its layers.

Just sitting there, intact.

Now imagine an inference engine that knows exactly how those weights relate to the mathematical operations required by the model.

It takes a bounded section of weights, brings the necessary data into active memory, performs the calculation, saves whatever mathematical continuation is needed, and releases the memory.

Then it moves to the next section.

And the next.

The total size of the model on SSD might be enormous.

But the active working memory could, in principle, remain much smaller.

I set an ambitious research target: approximately **150–160 MB of active RAM**.

Not 150 MB of model weights.

Not 150 MB of Python memory while conveniently ignoring everything else.

The whole active inference workload.

And I want it to be quick, too.

Which is where this becomes difficult, because repeatedly moving enormous amounts of data from SSD is expensive. A mathematically correct model that takes forever to respond isn't particularly helpful.

But difficult and impossible are not the same thing.

And I wanted to find out what the actual limits were.

## TRIP: the part that knows where things are

This wasn't happening in isolation.

I'd previously been developing something called TRIP — *Traversal & Retrieval through Internet Parsing*.

TRIP originally concerned navigating information: following sources, parsing them, comparing claims, dealing with contradictions and remembering useful paths through what can become an enormous network of knowledge.

Alongside that was QKE — *Quant Knows Everything* — which explored how to preserve useful convergence routes and evidence references rather than endlessly storing everything encountered.

The important part was the philosophy.

**Remember how something was resolved, but don't confuse remembering the route with proving the answer is still correct.**

Then it occurred to me.

Why should that kind of traversal apply only to information on the internet?

A neural network has a mathematical dependency structure.

Its operations have inputs, outputs and relationships. Its tensors have shapes, types and locations. Some models even have conditional branches determining which parts of the network are active.

Could TRIP traverse that?

Could it learn where the mathematics lives?

Could it prepare the execution paths before the model even starts answering questions?

And perhaps most interestingly:

**Could it do that while the model is still downloading?**

## Preparing a model before it arrives

This became TRIP Preflight.

Rather than simply download a huge collection of weights and hope the runtime sorts everything out afterwards, why not prepare the model as it arrives?

Safetensors checkpoints contain structural metadata describing tensors and their locations.

TRIP Preflight can inspect this information, verify files, catalogue the tensors, map dependencies and prepare legal execution schedules.

The engine then knows where the necessary mathematics can be found.

It can create seed PyThreads containing prepared routes.

Those aren't completed calculations. They don't know the future tokens someone will ask the model to generate.

They're starting maps.

Then, during inference, the active PyThreads take over.

The model performs its actual numerical operations, the mathematical continuation is preserved, and successful execution routes can be remembered.

An important distinction: this does not mean we can randomly skip a weight or an operation because we've visited something similar before.

If the original model mathematically requires an operation, we have to perform it, unless we've independently demonstrated a truly equivalent alternative.

The original model's mathematics remains the authority.

The point is to make accessing, scheduling, performing and resuming that mathematics more efficient.

## Then came Mixture-of-Experts

And this is where I got particularly excited.

Some language models are dense, meaning most of their network participates in processing each token.

Others use a Mixture-of-Experts architecture, or MoE.

An MoE model contains multiple expert networks, but only a selected subset might be activated for a particular token.

The selection is made by the model's learned router.

Now imagine that alongside TRIP.

The model determines which experts are mathematically required.

TRIP knows where those experts live on disk, how to load them, what dependencies precede them, and which loading schedules have worked previously.

PyThreads preserve the mathematical continuation.

ML-KTRIP executes the operations.

The model isn't becoming smaller.

We're attempting to make its original computation more accessible.

And suddenly the no-gate idea from that first little thought experiment begins to look rather different.

## So we built it

Eventually I stopped wanting to discuss the concept and started wanting to see whether it could actually do anything.

We began with small mathematical experiments.

Could a route be discovered and remembered?

Could a no-gate be validly certified?

Could a calculation be stopped and resumed?

Could disk-backed matrix operations produce the same results as a conventional reference calculation?

Then we moved to transformer-shaped computations.

A 24-layer Qwen-shaped architecture.

Grouped-query attention. Rotary positional embeddings. Normalisation. Feed-forward networks. Token-by-token mathematical continuation.

We tested it with synthetic weights and compared the resulting mathematics.

Then we built a 12-layer synthetic Mixture-of-Experts experiment.

TRIP Preflight prepared the routes before inference.

The MoE runtime selected experts and performed their mathematics.

The preflight and on-demand execution paths produced identical outputs to the reference in our tests.

Nineteen validation checks passed.

And we discovered something useful along the way: preparing those routes meant the runtime didn't need to rediscover expert locations during the tested generation sequence.

Did it make the model dramatically faster?

Not in those measurements.

But it demonstrated that the components could operate together.

That mattered.

Because the original idea was no longer just something I'd drawn on a piece of paper.

It was performing actual numerical operations.

## And somehow, there's now an application

This is probably the part I still find slightly ridiculous.

We built a desktop application.

For macOS and Windows.

With a model library, Safetensors downloads, TRIP preparation, PyThreads, conversations, and a little artefact studio where the model can produce code and HTML for inspection.

The interface is soft grey, lilac and off-white, because apparently building an experimental inference engine wasn't enough. It also had to be quite lovely.

And I wanted it to feel like somewhere you could actually work.

Not a developer terminal wearing a nicer coat.

A place where someone could download a model, chat, create things and understand what their own computer is doing.

Generated code doesn't get unrestricted access to your machine. Tool execution is intended to be explicitly approved and isolated.

The inference machinery lives behind the experience.

Which is exactly where I think it belongs.

Then came the experimental build.

And, rather wonderfully, it appears we've reached the point of natural-language inference.

I'm still establishing precisely which checkpoint operations, numerical comparisons, memory measurements and runtime paths have been validated in that latest build.

I don't want to confuse a working conversation with proving every research objective.

But after all the mathematics, the experiments, the errors and the revisions, seeing the engine begin to do the thing it was designed to do is quite something.

I genuinely can't quite believe it.

## What we haven't proven

I think this is important.

We've demonstrated real pieces of the architecture.

We have functioning experimental execution machinery.

But we have not yet independently established that a complete enormous pretrained model can operate with **150–160 MB of total physical RAM** at a genuinely useful conversational speed.

That's still the big test.

We also need rigorous pretrained-model comparisons, proper native-system measurements, security review and more work on the full TRIP traversal mechanics.

And there are already excellent inference systems exploring weight offloading, memory mapping and expert caching.

I'm not claiming that putting weights on an SSD is a new invention.

What I'm investigating is whether the combination of **bounded execution memory, mathematical continuation, remembered no-gate paths, checkpoint-time preparation and an inference engine that performs the model's operations itself** can produce something meaningfully different.

Something measurable.

Something independently reproducible.

Something useful.

I actually rather like that there's still so much to discover.

## What happens next?

I have a slightly outrageous long-term picture in my head.

Imagine having several complete models stored on your computer.

A Qwen model.

A Hermes model.

A Mistral model.

Perhaps an enormous Kimi.

Different architectures. Different capabilities. Different strengths.

And instead of requiring enough RAM to keep their weights resident, ML-KTRIP maintains a small computational environment and accesses each model independently when needed.

We're not at cross-model mathematical interoperability. That would be another research problem entirely.

For now, I want to prove one complete model at a time.

Correctly.

Efficiently.

With transparent measurements.

And, ideally, on ordinary computers.

Because I think access matters.

I don't particularly like the idea that the ability to experiment with powerful artificial intelligence should be reserved for people who can afford expensive specialist hardware.

If we can make that boundary smaller, that's worth working on.

And when we're ready to share the project more widely, I'd like to do that responsibly, through our *Quant Gives Back* licensing philosophy, with proper safety testing and a clear account of what the technology can and cannot do.

Not technology for technology's sake.

Something useful.

Something people can build upon.

## The part I keep coming back to

It really did begin with a very small idea.

A asks B a question.

B doesn't know.

B asks C.

C knows someone who can help.

The answer comes back.

And next time, we remember the journey.

That's all.

A domino of no-gates.

A tiny mathematical notebook.

An enormous collection of numbers sitting quietly on an SSD.

And a question about whether we could make them work together.

I have absolutely no idea how far this will go.

I don't know if we'll hit every target.

I don't know how the final architecture will compare with the best inference engines in the world.

But I know considerably more than I did when I first asked the question.

And we've actually built something.

Which, if I'm being perfectly truthful, is the bit that makes me happiest.

Sometimes you start with a question that sounds almost laughably simple.

Then you follow it.

Then you follow the next one.

And one daycan , somewhat unexpectedly, the mathematics starts answering back.

**That's ML-KTRIP.**

And I think we're only just beginning.

