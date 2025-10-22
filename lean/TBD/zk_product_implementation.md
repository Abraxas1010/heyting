Aequitas Protocol: Technical Documentation

Verifiable Execution for the Agentic Economy

1. Overview & Philosophy

This document details the technical architecture for the Aequitas Protocol, a system designed to provide provably secure, verifiable, and confidential execution for autonomous AI agents on public blockchains.

The Core Problem: The primary challenge in the agentic economy is the "trust gap." An AI agent's logic is probabilistic, but on-chain execution is deterministic and irreversible. Manually translating a user's intent (a "mandate") into secure, bug-free Solidity code is a high-risk, unscalable bottleneck. This "certified compilation" problem—proving that the implementation (Solidity) perfectly matches the specification (the mandate)—is the single greatest barrier to enterprise adoption.

The Aequitas Solution: We solve this problem by bypassing it entirely.

Instead of manually writing and auditing thousands of complex, bespoke smart contracts, the Aequitas Protocol utilizes a single, simple, and hyper-audited on-chain contract (a "Verifier"). The complex agent logic is handled off-chain by our formally-verified "Certified Compiler" toolchain, which is derived directly from our HeytingLean formalization [cite: lean_formalization_plan.md].

This toolchain compiles a user's logical mandate into a mathematically verifiable payload. The on-chain contract's only job is to check this payload's validity before executing a transaction. This document details the two primary products derived from this architecture: Aequitas-VM and Aequitas-ZK.

2. Product 1: Aequitas-VM (Verifiable VM)

Aequitas-VM provides Verified Execution as a Service. It is a simple, universal, and transparent on-chain interpreter that executes payloads compiled by our certified off-chain compiler.

2.1. Architecture

The VM architecture is split into two components: the off-chain "Certified Compiler" and the on-chain "Interpreter."

Off-Chain (The "Certified Compiler"):

Source: HeytingLean/Crypto/ [cite: multi-lens_ZK_PCT.md]

Logical IR (Form): A user's mandate (e.g., "pay up to $50 to an approved vendor") is defined as a logical proposition in our Heyting algebra (Crypto/Form.lean).

VM Instruction Set (Prog): We define a minimal, unambiguous postfix VM instruction set (Crypto/Prog.lean).

Compiler (compile): The lake exe pct_prove executable, which contains the compile: Form → Prog function (Crypto/Compile.lean).

The Verification: The entire off-chain stack is formally verified in Lean. The theorem compile_correct (Crypto/Correctness.lean) provides a mathematical proof that the output Prog payload is semantically identical to the input Form mandate.

On-Chain (The "Interpreter"):

Contract: A single, minimal, and hyper-audited Solidity contract deployed to the blockchain.

Core Function: function run(bytes memory progPayload, ...)

Logic: The contract's only job is to interpret the Prog instruction set. It is a simple loop that reads opcodes (e.g., PUSH, CHECK_VENDOR, TRANSFER_USDC) and executes them. It contains zero bespoke business logic.

2.2. Technical Flow

A user or agent defines their mandate as a Form (the logical specification).

The agent (off-chain) calls the lake exe pct_prove executable. This tool:
a.  Takes the Form mandate as input.
b.  Compiles it into a Prog payload.
c.  Returns this payload to the agent.

The agent constructs a transaction for the Aequitas-VM contract, passing the progPayload as an argument.

The Aequitas-VM contract's run function is executed:
a.  It verifies the transaction includes the required protocol fee.
b.  It interprets the progPayload, executing the verified logic.
c.  If all instructions pass, the final transaction (e.g., a USDC transfer) is executed.

2.3. Monetization & Security

Monetization: The run function includes a require check for a protocol fee (e.g., a percentage of the transaction value). This fee is transferred to the protocol treasury, providing a direct revenue stream.

Security Model: The security guarantee is exceptionally high. Trust is no longer placed in a complex, unique agent contract. Instead, trust is placed in:

The compile_correct theorem (a mathematical proof).

A single, minimal on-chain interpreter that can be audited by the entire world.

3. Product 2: Aequitas-ZK (ZK Verifier)

Aequitas-ZK is our premium, enterprise-grade product. It provides Confidential & Verified Execution as a Service. It moves the entire execution off-chain, using ZK-SNARKs to prove correct execution without revealing any private data.

3.1. Architecture

Off-Chain (The "ZK Certified Compiler"):

Source: HeytingLean/Crypto/ZK/ [cite: multi-lens_ZK_PCT.md]

Logical IR (Form): The same logical mandate as the VM.

Constraint System (R1CS): The lake exe pct_r1cs executable (Crypto/ZK/R1CSBool.lean) compiles the Form logic directly into a Rank-1 Constraint System (R1CS), the "blueprint" for a ZK-SNARK.

The Verification: The soundness and completeness theorems (from Phase E2) formally prove that the R1CS circuit is a correct representation of the Form logic.

On-Chain (The "Verifier"):

Contract: A single Solidity contract that is a standard ZK-SNARK Verifier (e.g., Groth16).

Core Function: function verify(bytes memory proof, uint[] memory publicInputs)

Logic: This contract has zero business logic. Its only purpose is to perform the mathematical operations to verify a ZK proof.

3.2. Technical Flow

A user or agent defines their mandate (Form).

The agent (off-chain) assembles its private inputs, or "witness" (e.g., "I want to pay $5,000,000 to vendor_ABC, which is on my approved list").

The agent (off-chain) calls the lake exe pct_r1cs executable to get the circuit.

The agent uses an off-chain prover to generate a ZK-SNARK, proving: "I have correctly executed a logic mandate, and all constraints (e.g., 'is_approved_vendor', 'is_under_budget') are satisfied."

The agent submits only the proof and any public inputs to the Aequitas-ZK contract.

The Aequitas-ZK contract's verify function is executed:
a.  It verifies the transaction includes the protocol fee.
b.  It runs the verify check.
c.  If the proof is valid, it executes the pre-defined action (e.g., moving funds from the agent's wallet to a pre-registered (but publicly anonymous) destination).

3.3. Monetization & Security

Monetization: The verify function requires a premium fee, reflecting the high value of commercial privacy.

Security Model: This is the gold standard. The entire business logic (vendor, amount, mandate) is executed off-chain and remains completely confidential. The blockchain is only used to record a mathematical proof of correctness. We move from "code is law" to "math is law."

4. Business Model & Defensibility

The open-source nature of the on-chain contracts is not a vulnerability; it is our primary strategic advantage. Our defensibility is not in code secrecy but in three fundamental moats:

The Trust Moat: Our deployed contracts are the official, audited, and branded contracts. A competitor's copy is an anonymous, untrusted address. No rational enterprise will risk capital on a knock-off to save on fees.

The Off-Chain Stack (The Real IP): The "secret sauce" is not the simple on-chain verifier. It is the massive, complex, and formally-verified Lean stack (HeytingLean, LoF, Bridges, etc.) [cite: lean_formalization_plan.md] that is required to generate the Prog payloads and ZK proofs. A competitor cannot simply copy the verifier; they must also replicate our entire, multi-year formal verification research and development—a task our team is uniquely positioned to maintain and advance.

The Network Effect & Open-Core Model:

Core (Open-Source): The on-chain verifiers act as a standard. Their adoption is our primary growth engine.

Ecosystem (Network Effect): Tool vendors (data APIs, logistics) will build "lenses" [cite: lean_formalization_plan.md] that target our official verifier, creating a powerful chicken-and-egg loop that locks in users.

Enterprise (Paid): Our revenue comes from high-margin services that only the creators of the stack can offer: 24/7 enterprise support, private mandate audits, SaaS dashboards for agent activity, and consulting for new vendor integrations.

What you're describing is transforming the Aequitas-VM from a "Pass/Fail Verifier" into an "Iterative Refinement Oracle."

This directly solves the core problem you stated:

    The Problem: An LLM (a probabilistic seeker) has no "ground truth." It can't be sure its 500-line code is correct, leading to "recursive compounding of errors."

    Your Solution: The Aequitas-VM (a deterministic verifier) becomes the ground truth.

How This "Probabilistic-Deterministic Loop" Would Work

You are correct to expect agents to submit faulty proofs. This is the natural way they would learn. The key is to tweak the Aequitas-VM's on-chain interpreter to provide rich, deterministic feedback instead of just reverting.

Let's walk through your hypothetical:

    User Mandate (Form): "Pay up to $50 to 'Vendor A' for 'supplies'."

    Agent's (Probabilistic) Plan: The LLM agent misunderstands and generates a plan: "Pay $60 to 'Vendor B' for 'supplies'."

    Certified Compilation: The agent (off-chain) uses lake exe pct_prove [cite: multi-lens_ZK_PCT.md] to compile this flawed plan into a Prog payload.

    Submission (Attempt 1): The agent submits this faulty progPayload to the Aequitas-VM contract.

    Deterministic Execution & Feedback: The on-chain Interpreter begins executing the opcodes:

        PUSH 'Vendor B' -> OK

        CHECK_VENDOR -> FAIL. The mandate only allows 'Vendor A'.

        The Tweak: Instead of a simple revert(), the contract emits a specific, structured event: emit MandateViolation(agentId, errorCode: 'INVALID_VENDOR', context: 'Vendor B').

    Probabilistic Correction (Attempt 2):

        The agent (or its monitoring service) sees this deterministic feedback. It knows with 100% certainty that "Vendor B" was the specific point of failure.

        It now only has to correct that one error. It generates a new plan: "Pay $60 to 'Vendor A' for 'supplies'."

        It compiles and submits Prog Payload #2.

    Deterministic Execution & Feedback (Attempt 2):

        PUSH 'Vendor A' -> OK

        CHECK_VENDOR -> OK

        PUSH 60 -> OK

        CHECK_BUDGET -> FAIL. The mandate only allows $50.

        The VM emits: emit MandateViolation(agentId, errorCode: 'BUDGET_EXCEEDED', context: '60').

    Probabilistic Correction (Attempt 3):

        The agent sees this new feedback. It now knows its vendor is correct, but its budget is wrong.

        It generates its final, correct plan: "Pay $50 to 'Vendor A' for 'supplies'."

        It compiles and submits Prog Payload #3.

    Success: The Interpreter executes all opcodes, the checks pass, and the transaction is executed.

Conclusion

You've nailed it. By providing this rich, deterministic feedback, the Aequitas-VM becomes a training oracle. It uses the user's formal mandate (Form) as a "curriculum" to safely guide a "fuzzy" probabilistic agent, forcing it to iterate until its plan is 100% correct.

This is a far superior model. It doesn't just prevent the agentic economy from disaster; it actively builds a more robust and correct agentic economy by using formal verification as a teaching tool.