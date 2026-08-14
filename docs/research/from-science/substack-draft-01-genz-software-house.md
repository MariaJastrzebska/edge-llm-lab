# [SZKIC #1 na Substacka — seria „Gen Z software house"]
# My software house lives in my phone

A year ago I started this newsletter to write about small language models.
Then I stopped — not because I ran out of things to say, but because I felt
I wasn't "enough" to say them. Half a year of posting short videos on TikTok
cured me of that. You will never be good enough for everyone, and that turns
out to be liberating. So: I'm back, and this time I'm writing about the thing
I actually do all day — running what I can only describe as a one-woman
software house, from my phone, mostly by voice, usually on a walk.

Here is what that looked like this week, concretely.

An issue lands in my CV-assistant app — a user's pasted job offer gets lost
in the chat. I don't open a laptop. My Telegram bot writes to me, in plain
language, three layers deep: first what the user experiences, then what it
proposes to do about it, then — because I asked it to teach me — a short
technical section that explains the how, with every term unpacked. I listen
to it as a voice note while walking along the river. I answer with a voice
note of my own: "zatwierdzam" — approved. The agent writes the fix, puts it
on a staging copy of the app, and waits. Nothing goes to production until I
say, in plain Polish, "deploy to production." One ticket at a time, because I
told it that a pile of parallel plans makes "approved" ambiguous — and it
remembered.

Meanwhile, at night, my old MacBook Air runs physics — well, measurement.
I'm chasing a research question about how much energy it costs to force a
language model to speak in a strict format (it matters more than you'd think
once models run on devices instead of datacenters). The experiment queue
runs while I sleep. In the morning I get the night report as a voice note —
the WHOLE report, findings first, deltas spelled out, split into parts so
nothing gets cut off. I complained exactly once about a truncated message;
the pipeline has never cut me off since.

The unglamorous truth is that none of this needed exotic tools. A Linux box
that costs less than a gym membership. Telegram bots — one per project,
because I got cross when reports mixed. Free text-to-speech. Cron. Git. One
AI subscription doing the thinking, and a growing pile of plain-Markdown
"constitution" files that encode how I want to work: which decisions are
mine (there's literally a file called NEEDS_INPUT.md — when it exists, the
agents stop and wait for me), what a report must sound like (conclusions
first, full sentences, teach me the technical parts), what an experiment
must record to count (hardware manifest, paired runs, honest nulls).

That last part is the real lesson, and it's the thesis of this series: the
code my agents write is disposable; the *specification of how I work* is
not. Every hour I spent saying "no — one ticket at a time", "no — explain
it like I'm smart but new", "no — never present me a closed ticket" compiled
into files that any future model will read and obey. My generation is
supposed to be glued to phones. Fine. I've made the phone the management
floor, the laptop a night-shift lab, and the basement — empty. I'd rather
be walking.

Next time: the constitution files themselves — how a few hundred lines of
Markdown make an AI wait for a human.

*(Szkic do Twojej redakcji — obetnij, przestaw, dopisz swój głos. Fakty są
z naszej realnej historii z tego tygodnia.)*
