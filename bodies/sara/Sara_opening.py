import time
import stomp
import json

hosts = [("localhost", 61613)]
conn = stomp.Connection(host_and_ports=hosts, auto_content_length=False)
conn.connect("admin", "admin", wait=True)

headers = dict()
headers["ELVISH_SCOPE"] = "DEFAULT_SCOPE"
headers["MESSAGE_PREFIX"] = "vrWEFUI"


body3 = "vrWEFUI start"
headers["MESSAGE_PREFIX"] = "vrExpress"
conn.send(
    body=body3, destination="/topic/DEFAULT_SCOPE", headers=headers, persistent=True
)


# body1 = 'vrSpeak Brad user 1000037 <?xml version="1.0" encoding="utf-16"?><act><participant id="brad" role="actor" /><bml><speech id="sp1" ref="voice_defaultTTS" type="application/ssml+xml"><mark name="T1" />hi <mark name="T2" /><mark name="T2" />i am <mark name="T3" /><mark name="T3" />sara <mark name="T4" /><mark name="T4" />what is <mark name="T5" /><mark name="T5" />your <mark name="T6" /><mark name="T6" />name <mark name="T7" /></speech><event message="vrAgentSpeech partial 1480100645859 T1 hi " stroke="sp1:T1" /><event message="vrAgentSpeech partial 1480100645859 T3 hi i am " stroke="sp1:T3" /><event message="vrAgentSpeech partial 1480100645859 T5 hi i am sara " stroke="sp1:T5" /><event message="vrAgentSpeech partial 1480100645859 T7 hi i am sara what is " stroke="sp1:T7" /><event message="vrAgentSpeech partial 1480100645859 T9 hi i am sara what is your " stroke="sp1:T9" /><event message="vrAgentSpeech partial 1480100645859 T11 hi i am sara what is your name " stroke="sp1:T11" /><sbm:event message="vrSpoke brad user 1480100642410 hi i am sara what is your name." stroke="sp1:relax" xmlns:xml="http://www.w3.org/XML/1998/namespace" xmlns:sbm="http://ict.usc.edu" /><intonation_break start="sp1:T0" end="sp1:T3"/><intonation_tone endtone="L-H%" start="sp1:T0" end="sp1:T4"/><intonation_tone endtone="L-L%" start="sp1:T3" end="sp1:T4"/><face type="FACS" au="2" amount="5.0"/><animation stroke="sp1:T3" priority="1" name="beat_low_right_Sara"/><head type="NOD" amount="0.1" repeats="1" relax="sp1:T3" priority="1" /><face type="FACS" stroke="sp1:T5" au="101" amount="1.0" priority="1" /><face type="FACS" start="sp1:T0" end="sp1:T0" au="103" amount="0.5"/><animation start="sp1:T0" priority="1" name="beat_middle_right_Sara"/><gaze participant = "Brad" target="Front" angle="0" start="7" sbm:joint-range="EYES" xmlns:sbm="http://ict.usc.edu"/></bml></act>'
# # body2 = 'vrSpeak Brad User 1466989728367 <?xml version="1.0" encoding="utf-16"?><act><participant id="Brad" role="actor" /><bml><event message="vrSpoke Brad User 6 " xmlns:sbm="​http://ict.usc.edu​" /> <animation priority="1" name="beat_low_right_sara" /></bml></act>'
# headers["MESSAGE_PREFIX"] = "vrSpeak"
# conn.send(
#     body=body1, destination="/topic/DEFAULT_SCOPE", headers=headers, persistent=True
# )


# time.sleep(5)
# # body1 = 'vrSpeak Brad user 1000037 <?xml version="1.0" encoding="utf-16"?><act><participant id="brad" role="actor" /><bml><speech id="sp1" ref="voice_defaultTTS" type="application/ssml+xml"><mark name="T1" />hi <mark name="T2" /><mark name="T2" />i am <mark name="T3" /><mark name="T3" />sara <mark name="T4" /><mark name="T4" />what is <mark name="T5" /><mark name="T5" />your <mark name="T6" /><mark name="T6" />name <mark name="T7" /></speech><event message="vrAgentSpeech partial 1480100645859 T1 hi " stroke="sp1:T1" /><event message="vrAgentSpeech partial 1480100645859 T3 hi i am " stroke="sp1:T3" /><event message="vrAgentSpeech partial 1480100645859 T5 hi i am sara " stroke="sp1:T5" /><event message="vrAgentSpeech partial 1480100645859 T7 hi i am sara what is " stroke="sp1:T7" /><event message="vrAgentSpeech partial 1480100645859 T9 hi i am sara what is your " stroke="sp1:T9" /><event message="vrAgentSpeech partial 1480100645859 T11 hi i am sara what is your name " stroke="sp1:T11" /><sbm:event message="vrSpoke brad user 1480100642410 hi i am sara what is your name." stroke="sp1:relax" xmlns:xml="http://www.w3.org/XML/1998/namespace" xmlns:sbm="http://ict.usc.edu" /><intonation_break start="sp1:T0" end="sp1:T3"/><intonation_tone endtone="L-H%" start="sp1:T0" end="sp1:T4"/><intonation_tone endtone="L-L%" start="sp1:T3" end="sp1:T4"/><face type="FACS" au="2" amount="5.0"/><animation stroke="sp1:T3" priority="1" name="beat_low_right_Sara"/><head type="NOD" amount="0.1" repeats="1" relax="sp1:T3" priority="1" /><face type="FACS" stroke="sp1:T5" au="101" amount="1.0" priority="1" /><face type="FACS" start="sp1:T0" end="sp1:T0" au="103" amount="0.5"/><animation start="sp1:T0" priority="1" name="beat_middle_right_Sara"/><gaze participant = "Brad" target="Front" angle="0" start="7" sbm:joint-range="EYES" xmlns:sbm="http://ict.usc.edu"/></bml></act>'
# # body2 = 'vrSpeak Brad User 1466989728367 <?xml version="1.0" encoding="utf-16"?><act><participant id="Brad" role="actor" /><bml><event message="vrSpoke Brad User 6 " xmlns:sbm="​http://ict.usc.edu​" /> <animation priority="1" name="beat_low_right_sara" /></bml></act>'
# body4 = """vrExpress Brad user 1480100642410 <?xml version="1.0" encoding="UTF-8" standalone="no" ?>
#         <act> <participant id="Brad" role="actor" /><fml>
#         <turn continuation="false" />
#         <affect type="neutral" target="addressee">
#         </affect> <culture type="neutral"> </culture>
#         <personality type="neutral"> </personality>
#         <sentence phase="greetings" intention="greeting" strategy="NONE" rapport="NONE" text="Hi, I'm Sara. What's your name?" />
#         </fml>
#         <bml>
#         <speech>
#         Hi I'm Sara What's your name?</speech>
#         </bml>
#         <ssml>
#         <speech><s>Hi <break strength="medium"/> I'm Sara <break strength="medium"/> What's <prosody pitch="high">your</prosody> name?</s></speech>
#         </ssml>
#         </act>"""
# headers["MESSAGE_PREFIX"] = "vrExpress"
# conn.send(
#     body=body4, destination="/topic/DEFAULT_SCOPE", headers=headers, persistent=True
# )

# print("body4")

body6 = """Brad user 1725288162673 <?xml version="1.0" encoding="utf-16"?>
<act>
<participant id="brad" role="actor" /> <bml>
<speech id="sp1" ref="voice_defaultTTS" type="application/ssml xml">
<mark name="T1" />you <mark name="T2" />
</speech>
<event message="vrAgentSpeech partial 1725288162673 T1 you " stroke="sp1:T1" />
<sbm:event message="vrSpoke brad user 1337363228078-9-1 you." stroke="sp1:relax" xmlns:xml="http://www.w3.org/XML/1998/namespace" xmlns:sbm="http://ict.usc.edu" />
<face type="FACS" stroke="sp1:T0" au="105" amount="0.5"  priority="1"/>
<animation start="sp1:T5" priority="1" name="world_economic_forum"/>
<intonation_break start="sp1:T1" end="sp1:T1"/>
</bml>
</act>"""

# body5 = """vrSpeak Brad user 1000037 <?xml version="1.0" encoding="utf-16"?>
# <act>
# <participant id="brad" role="actor" /> <bml>
# <speech id="sp1" ref="voice_defaultTTS" type="application/ssml+xml">
# <mark name="T1" />hi <mark name="T2" />
# <mark name="T2" />i'm <mark name="T3" />
# <mark name="T3" />sara <mark name="T4" />
# <mark name="T4" />what's <mark name="T5" />
# <mark name="T5" />your <mark name="T6" />
# <mark name="T6" />name <mark name="T7" />
# </speech>
# <event message="vrAgentSpeech partial 1725270663312 T1 hi " stroke="sp1:T1" />
# <event message="vrAgentSpeech partial 1725270663312 T3 hi i'm " stroke="sp1:T3" />
# <event message="vrAgentSpeech partial 1725270663312 T5 hi i'm sara " stroke="sp1:T5" />
# <event message="vrAgentSpeech partial 1725270663312 T7 hi i'm sara what's " stroke="sp1:T7" />
# <event message="vrAgentSpeech partial 1725270663312 T9 hi i'm sara what's your " stroke="sp1:T9" />
# <event message="vrAgentSpeech partial 1725270663312 T11 hi i'm sara what's your name " stroke="sp1:T11" />
# <sbm:event message="vrSpoke brad user 1337363228078-9-1 hi i'm sara what's your name." stroke="sp1:relax" xmlns:xml="http://www.w3.org/XML/1998/namespace" xmlns:sbm="http://ict.usc.edu" />
# <face type="FACS" stroke="sp1:T0" au="105" amount="0.5"  priority="1"/>
# <animation start="sp1:T5" priority="1" name="world_economic_forum"/>
# <intonation_break start="sp1:T7" end="sp1:T7"/>
# <intonation_break start="sp1:T7" end="sp1:T7"/>
# <face type="FACS" stroke="sp1:T0" au="105" amount="0.25"  priority="1"/>
# <face type="FACS" stroke="sp1:T0" au="106" amount="0.5"  priority="1"/>
# <face type="FACS" stroke="sp1:T0" au="2" amount="0.5"  priority="1"/>
# </bml>
# </act>
# """
# headers["MESSAGE_PREFIX"] = "vrSpeak"
# conn.send(
#     body=body5, destination="/topic/DEFAULT_SCOPE", headers=headers, persistent=True
# )

# time.sleep(5)
# body1 = 'vrSpeak Brad user 1000037 <?xml version="1.0" encoding="utf-16"?><act><participant id="brad" role="actor" /><bml><speech id="sp1" ref="voice_defaultTTS" type="application/ssml+xml"><mark name="T1" />hi <mark name="T2" /><mark name="T2" />i am <mark name="T3" /><mark name="T3" />sara <mark name="T4" /><mark name="T4" />what is <mark name="T5" /><mark name="T5" />your <mark name="T6" /><mark name="T6" />name <mark name="T7" /></speech><event message="vrAgentSpeech partial 1480100645859 T1 hi " stroke="sp1:T1" /><event message="vrAgentSpeech partial 1480100645859 T3 hi i am " stroke="sp1:T3" /><event message="vrAgentSpeech partial 1480100645859 T5 hi i am sara " stroke="sp1:T5" /><event message="vrAgentSpeech partial 1480100645859 T7 hi i am sara what is " stroke="sp1:T7" /><event message="vrAgentSpeech partial 1480100645859 T9 hi i am sara what is your " stroke="sp1:T9" /><event message="vrAgentSpeech partial 1480100645859 T11 hi i am sara what is your name " stroke="sp1:T11" /><sbm:event message="vrSpoke brad user 1480100642410 hi i am sara what is your name." stroke="sp1:relax" xmlns:xml="http://www.w3.org/XML/1998/namespace" xmlns:sbm="http://ict.usc.edu" /><intonation_break start="sp1:T0" end="sp1:T3"/><intonation_tone endtone="L-H%" start="sp1:T0" end="sp1:T4"/><intonation_tone endtone="L-L%" start="sp1:T3" end="sp1:T4"/><face type="FACS" au="2" amount="5.0"/><animation stroke="sp1:T3" priority="1" name="beat_low_right_Sara"/><head type="NOD" amount="0.1" repeats="1" relax="sp1:T3" priority="1" /><face type="FACS" stroke="sp1:T5" au="101" amount="1.0" priority="1" /><face type="FACS" start="sp1:T0" end="sp1:T0" au="103" amount="0.5"/><animation start="sp1:T0" priority="1" name="beat_middle_right_Sara"/><gaze participant = "Brad" target="Front" angle="0" start="7" sbm:joint-range="EYES" xmlns:sbm="http://ict.usc.edu"/></bml></act>'
# # body2 = 'vrSpeak Brad User 1466989728367 <?xml version="1.0" encoding="utf-16"?><act><participant id="Brad" role="actor" /><bml><event message="vrSpoke Brad User 6 " xmlns:sbm="​http://ict.usc.edu​" /> <animation priority="1" name="beat_low_right_sara" /></bml></act>'
# headers["MESSAGE_PREFIX"] = "vrSpeak"
# conn.send(
#     body=body1, destination="/topic/DEFAULT_SCOPE", headers=headers, persistent=True
# )
