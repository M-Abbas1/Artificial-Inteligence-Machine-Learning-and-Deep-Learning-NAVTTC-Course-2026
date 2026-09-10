# GRU Worked Example: “Sara was happy because she won”

> **Teaching goal:** Understand how a GRU processes a sentence one word at a time, uses the **Reset Gate** to control previous information, uses the **Update Gate** to control memory replacement, creates a **Candidate Hidden State**, and produces the **Final Hidden State**.

---

## 1. Our Sentence

We will use:

```text
Sara → was → happy → because → she → won
```

We assume a GRU with:

- **Hidden size = 4**
- Therefore, the GRU's memory is a **4-dimensional vector**.

For teaching purposes, imagine:

```text
h = [ gender, number, person, tense ]
```

### Important clarification

This is a **teaching analogy**.

In a real GRU, the individual dimensions of the hidden state do **not necessarily correspond to explicit concepts** such as gender, number, person, or tense. Neural networks usually learn distributed representations.

However, imagining four dimensions this way makes the mechanism much easier to visualize.

---

# 2. Start with No Memory

Before processing the first word:

```math
h_0 = [0, 0, 0, 0]
```

Our GRU starts with an empty memory.

```text
Memory:

[ 0    0    0    0 ]
  ↓    ↓    ↓    ↓
gender number person tense
```

Think of `h₀` as the model's memory before it has read anything.

---

# 3. Word 1: “Sara”

The word `"Sara"` is converted into an input vector.

For simplicity:

```math
x_1 = [0.8, 0.7, 0.9, 0.1]
```

The GRU processes this input together with the previous hidden state:

```text
Current word              Previous memory
     ↓                         ↓
    Sara                      h₀
     │                         │
     └──────────┬──────────────┘
                ↓
               GRU
                ↓
               h₁
```

For our simplified example, suppose the resulting hidden state is:

```math
h_1 = [0.90, 0.80, 0.90, 0.10]
```

So after seeing **Sara**:

```text
             Sara
              ↓
h₁ = [0.90, 0.80, 0.90, 0.10]
       ↓      ↓      ↓      ↓
    gender  number  person  tense
```

The important idea:

> The GRU has created a memory about what it has seen so far.

---

# 4. Word 2: “was”

Now:

```math
x_2 = "was"
```

The GRU receives two things:

```text
Current word              Previous memory
     ↓                         ↓
    "was"                     h₁
     │                         │
     └──────────┬──────────────┘
                ↓
               GRU
                ↓
               h₂
```

Suppose:

```math
h_2 = [0.88, 0.79, 0.88, 0.65]
```

Notice that the **tense-related value** has changed considerably because `"was"` provides strong information about past tense.

The other information is still largely preserved.

---

# 5. Word 3: “happy”

Now:

```math
x_3 = "happy"
```

The GRU receives:

```math
x_3,\;h_2
```

and produces:

```math
h_3 = [0.87, 0.78, 0.87, 0.67]
```

The important point:

> The previous information has not simply disappeared. The GRU continues carrying information forward.

---

# 6. Word 4: “because”

Similarly:

```math
x_4 = "because"
```

Suppose:

```math
h_4 = [0.86, 0.78, 0.87, 0.67]
```

So after processing:

```text
Sara → was → happy → because
```

our memory is approximately:

```math
h_4 = [0.86, 0.78, 0.87, 0.67]
```

Now comes the interesting word:

# 7. Word 5: “she”

This is where we demonstrate the **actual GRU architecture**.

The GRU receives:

```math
x_5 = "she"
```

and the previous hidden state:

```math
h_4 = [0.86, 0.78, 0.87, 0.67]
```

Conceptually:

```text
                    "she"
                      │
                      ▼
                 ┌────────┐
h₄ ─────────────→│  GRU   │
                 └────────┘
                      │
                      ▼
                     h₅
```

---

# 8. Reset Gate

The Reset Gate is:

```math
r_t = σ(W_r x_t + U_r h_{t-1} + b_r)
```

For our simplified example, suppose the calculation gives:

```math
\boxed{
r_5 = [0.95,\;0.90,\;0.95,\;0.20]
}
```

Visualize it:

```text
                 RESET GATE

        [ 0.95   0.90   0.95   0.20 ]
           ↓      ↓      ↓      ↓
        gender  number  person  tense
```

Ask students:

### What does 0.95 mean?

It means:

> **Keep/use a lot of the old information in this dimension.**

### What does 0.20 mean?

It means:

> **Use very little of the old information in this dimension.**

The Reset Gate therefore controls how much of the **previous hidden state** is used while constructing the candidate memory.

---

# 9. Apply the Reset Gate

The GRU performs element-wise multiplication:

```math
r_t \odot h_{t-1}
```

Therefore:

```math
[0.95,\;0.90,\;0.95,\;0.20]
\odot
[0.86,\;0.78,\;0.87,\;0.67]
```

Calculate each dimension:

```text
0.95 × 0.86  = 0.817
0.90 × 0.78  = 0.702
0.95 × 0.87  = 0.8265
0.20 × 0.67  = 0.134
```

Therefore:

```math
\boxed{
r_5 \odot h_4 =
[0.817,\;0.702,\;0.8265,\;0.134]
}
```

Visual representation:

```text
Old memory:

[0.86    0.78    0.87    0.67]

Reset gate:

[0.95    0.90    0.95    0.20]

             ↓ element-wise ×

Filtered old memory:

[0.817   0.702   0.8265  0.134]
```

### Teaching point

> **The Reset Gate controls how much previous memory participates in creating the candidate memory.**

---

# 10. Candidate Hidden State

Now the GRU creates a **Candidate Hidden State**.

The equation is:

```math
\tilde{h}_t =
\tanh(
W_hx_t +
U_h(r_t \odot h_{t-1}) +
b_h
)
```

For our simplified example, suppose the calculation gives:

```math
\boxed{
\tilde{h}_5 =
[0.95,\;0.85,\;0.92,\;0.30]
}
```

Think of this as:

> “If I were to create a new memory based on the current word and the relevant old information, this is what the memory could look like.”

```text
Candidate Memory

[ 0.95   0.85   0.92   0.30 ]
    ↓      ↓      ↓      ↓
 gender number person  tense
```

### Important idea

The candidate hidden state is **not yet the final memory**.

The GRU still has to decide:

> How much of this candidate should actually enter the memory?

That decision is made by the **Update Gate**.

---

# 11. Update Gate

The Update Gate is:

```math
z_t = σ(W_z x_t + U_z h_{t-1} + b_z)
```

Suppose:

```math
\boxed{
z_5 = [0.30,\;0.20,\;0.25,\;0.80]
}
```

Visualize it:

```text
UPDATE GATE

[ 0.30   0.20   0.25   0.80 ]
    ↓      ↓      ↓      ↓
 gender number person  tense
```

Interpret each value:

### Gender

```math
z = 0.30
```

Only **30%** candidate/new information is used.

### Number

```math
z = 0.20
```

Only **20%** candidate/new information is used.

### Person

```math
z = 0.25
```

Only **25%** candidate/new information is used.

### Tense

```math
z = 0.80
```

**80%** candidate/new information is used.

So, in our simplified analogy, the GRU is saying:

> “I mostly already know the gender, number, and person. I don't need to change them much.”

And:

> “I am more willing to update the tense information.”

---

# 12. Final Hidden State

Now we use the most important GRU equation:

```math
h_t =
(1-z_t) \odot h_{t-1}
+
z_t \odot \tilde{h}_t
```

This is where the GRU combines:

**Old Memory + Candidate Memory**

---

## Step A — Calculate `1 - z_t`

We have:

```math
z_5 = [0.30,\;0.20,\;0.25,\;0.80]
```

Therefore:

```math
1-z_5 = [0.70,\;0.80,\;0.75,\;0.20]
```

---

## Step B — Old Memory Contribution

Calculate:

```math
(1-z_5)\odot h_4
```

Substitute the values:

```math
=
[0.70,\;0.80,\;0.75,\;0.20]
\odot
[0.86,\;0.78,\;0.87,\;0.67]
```

Element-wise multiplication:

```text
0.70 × 0.86 = 0.602
0.80 × 0.78 = 0.624
0.75 × 0.87 = 0.6525
0.20 × 0.67 = 0.134
```

Therefore:

```math
\boxed{
(1-z_5)\odot h_4 =
[0.602,\;0.624,\;0.6525,\;0.134]
}
```

---

## Step C — New Memory Contribution

Now calculate:

```math
z_5 \odot \tilde{h}_5
```

Substitute:

```math
=
[0.30,\;0.20,\;0.25,\;0.80]
\odot
[0.95,\;0.85,\;0.92,\;0.30]
```

Element-wise multiplication:

```text
0.30 × 0.95 = 0.285
0.20 × 0.85 = 0.170
0.25 × 0.92 = 0.230
0.80 × 0.30 = 0.240
```

Therefore:

```math
\boxed{
z_5\odot\tilde{h}_5 =
[0.285,\;0.170,\;0.230,\;0.240]
}
```

---

# 13. Add the Two Contributions

Now:

```math
h_5 =
[0.602,\;0.624,\;0.6525,\;0.134]
+
[0.285,\;0.170,\;0.230,\;0.240]
```

Therefore:

```math
\boxed{
h_5 =
[0.887,\;0.794,\;0.8825,\;0.374]
}
```

🎯 **This is our new memory after processing the word “she”.**

---

# 14. Understand What Just Happened

The entire calculation can be summarized as:

```text
                  Current word
                      "she"
                        │
                        ▼
               ┌────────────────┐
Previous       │                │
memory h₄ ────→│      GRU       │
               │                │
               └───────┬────────┘
                       │
              ┌────────┴────────┐
              ↓                 ↓
        RESET GATE        UPDATE GATE
              │                 │
              ↓                 ↓
      Filter old memory    Control update
              │                 │
              ↓                 │
       Candidate memory          │
              │                 │
              └────────┬────────┘
                       ↓
                 Final memory
                       │
                       ↓
           h₅ = [0.887, 0.794,
                  0.8825, 0.374]
```

---

# 15. Then Process “won”

The next word is:

```text
she → won
```

The GRU receives:

```math
x_6 = "won"
```

and the memory from the previous step:

```math
h_5 =
[0.887,\;0.794,\;0.8825,\;0.374]
```

The same process happens again:

```text
x₆ + h₅
    ↓
Reset Gate
    ↓
Candidate Memory
    ↓
Update Gate
    ↓
Final Hidden State h₆
```

We don't need to perform all the arithmetic again in class.

The key idea is:

```text
Sara
  ↓
was
  ↓
happy
  ↓
because
  ↓
she
  ↓
won
  ↓
Final representation
```

Information originating from **Sara** can continue influencing later hidden states.

---

# 16. Reset Gate vs Update Gate

This is one of the most important concepts for students.

Students frequently confuse these two gates.

## Reset Gate

Ask:

> **“How much of the old memory should I use when creating the candidate?”**

It controls:

```math
r_t \odot h_{t-1}
```

Think:

```text
RESET GATE
     ↓
“Should I look at my old notes
 while writing a new note?”
```

---

## Update Gate

Ask:

> **“How much should I replace/update my current memory with the candidate?”**

It controls:

```math
(1-z_t)h_{t-1}
+
z_t\tilde{h}_t
```

Think:

```text
UPDATE GATE
     ↓
“How much of my old note should
 I replace with the new note?”
```

### Easy way to remember

> **Reset = controls old information used for the candidate.**

> **Update = controls how much candidate information enters the final memory.**

---

# 17. Complete GRU Calculation at a Glance

For the word **“she”**:

### Previous memory

```math
h_4=[0.86,\;0.78,\;0.87,\;0.67]
```

### Current input

```math
x_5=\text{"she"}
```

### Reset Gate

```math
r_5=[0.95,\;0.90,\;0.95,\;0.20]
```

### Filtered old memory

```math
r_5\odot h_4
=
[0.817,\;0.702,\;0.8265,\;0.134]
```

### Candidate memory

```math
\tilde h_5=
[0.95,\;0.85,\;0.92,\;0.30]
```

### Update Gate

```math
z_5=[0.30,\;0.20,\;0.25,\;0.80]
```

### Old contribution

```math
(1-z_5)\odot h_4
=
[0.602,\;0.624,\;0.6525,\;0.134]
```

### New contribution

```math
z_5\odot\tilde h_5
=
[0.285,\;0.170,\;0.230,\;0.240]
```

### Final memory

```math
\boxed{
h_5=[0.887,\;0.794,\;0.8825,\;0.374]
}
```

---

# 18. The Complete GRU Architecture

At any time step `t`, the process is:

```text
                         xₜ
                          │
                          ▼
              ┌────────────────────┐
              │                    │
hₜ₋₁ ────────→│       GRU          │
              │                    │
              └─────────┬──────────┘
                        │
              ┌─────────┴─────────┐
              ↓                   ↓
         Reset Gate           Update Gate
              │                   │
              ↓                   │
       rₜ ⊙ hₜ₋₁                  │
              │                   │
              ↓                   │
       Candidate h̃ₜ              │
              │                   │
              └────────┬──────────┘
                       ↓
                  Final hₜ
```

The four core equations are:

### Reset Gate

```math
\boxed{
r_t=\sigma(W_rx_t+U_rh_{t-1}+b_r)
}
```

### Candidate Hidden State

```math
\boxed{
\tilde{h}_t=
\tanh(
W_hx_t+
U_h(r_t\odot h_{t-1})+
b_h
)
}
```

### Update Gate

```math
\boxed{
z_t=\sigma(W_zx_t+U_zh_{t-1}+b_z)
}
```

### Final Hidden State

```math
\boxed{
h_t=
(1-z_t)\odot h_{t-1}
+
z_t\odot\tilde{h}_t
}
```

---

# 19. What Do the Activation Functions Do?

You can briefly explain these while teaching the equations.

## Sigmoid

Both gates use sigmoid:

```math
\sigma(x)=\frac{1}{1+e^{-x}}
```

It produces values between:

```text
0 ─────────────────── 1
```

So a gate can behave like a **soft switch**.

```text
0     → ignore / very little
0.5   → partial
1     → keep / use a lot
```

## Tanh

The candidate uses `tanh`:

```math
\tanh(x)
```

Its output lies between:

```text
-1 ───────── 0 ───────── +1
```

It helps keep the candidate hidden-state values bounded.

---

# 20. Final Summary Slide

### GRU = Controlled Memory

```text
                  Current input xₜ
                         │
                         ▼
                 ┌───────────────┐
                 │               │
Previous ───────→│     RESET     │
memory hₜ₋₁      │      GATE     │
                 └───────┬───────┘
                         │
                         ▼
                 Candidate h̃ₜ
                         │
                         │
Previous ────────────────┤
memory hₜ₋₁             │
                         ▼
                 ┌───────────────┐
                 │    UPDATE     │
                 │     GATE      │
                 └───────┬───────┘
                         │
                         ▼
                    Final hₜ
```

### Remember:

> **Reset Gate:** How much old information should I use to create the candidate?

> **Update Gate:** How much should I keep from the old memory versus take from the candidate?

> **Final Hidden State:** A controlled combination of old memory and new candidate information.

---

# 21. A Good Classroom Closing Question

After finishing the example, ask students:

> **“If the Update Gate has a value close to 0, will the new hidden state mostly resemble the old memory or the candidate memory?”**

Answer:

**Old memory.**

Because:

```math
h_t=(1-z_t)h_{t-1}+z_t\tilde h_t
```

If:

```math
z_t ≈ 0
```

then:

```math
h_t ≈ h_{t-1}
```

Conversely, if:

```math
z_t ≈ 1
```

then:

```math
h_t ≈ \tilde h_t
```

This is a very intuitive way to make the Update Gate stick in their minds.

---

## Final takeaway

The GRU processes the sentence **sequentially**, maintaining a hidden-state vector as its memory:

```text
x₁ → h₁
x₂ + h₁ → h₂
x₃ + h₂ → h₃
x₄ + h₃ → h₄
x₅ + h₄ → h₅
x₆ + h₅ → h₆
```

At every step:

```text
             Previous memory
                    │
                    ▼
              ┌───────────┐
Current input →│    GRU    │
              └─────┬─────┘
                    │
                    ▼
             New memory
```

**The GRU's intelligence is in deciding what information to carry forward, what information to reset, and what new information to write into its hidden state.**
