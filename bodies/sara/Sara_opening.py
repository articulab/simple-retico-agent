import stomp    
import json

hosts = [('localhost', 61613)]
conn = stomp.Connection(host_and_ports=hosts, auto_content_length=False)
conn.connect("admin", "admin", wait=True)

headers = dict()
headers['ELVISH_SCOPE'] = 'DEFAULT_SCOPE'
headers['MESSAGE_PREFIX'] = 'vrWEFUI'

 
body1 = 'vrSpeak Brad user 1000037 <?xml version="1.0" encoding="utf-16"?><act><participant id="brad" role="actor" /><bml><speech id="sp1" ref="voice_defaultTTS" type="application/ssml+xml"><mark name="T1" />hi <mark name="T2" /><mark name="T2" />i am <mark name="T3" /><mark name="T3" />sara <mark name="T4" /><mark name="T4" />what is <mark name="T5" /><mark name="T5" />your <mark name="T6" /><mark name="T6" />name <mark name="T7" /></speech><event message="vrAgentSpeech partial 1480100645859 T1 hi " stroke="sp1:T1" /><event message="vrAgentSpeech partial 1480100645859 T3 hi i am " stroke="sp1:T3" /><event message="vrAgentSpeech partial 1480100645859 T5 hi i am sara " stroke="sp1:T5" /><event message="vrAgentSpeech partial 1480100645859 T7 hi i am sara what is " stroke="sp1:T7" /><event message="vrAgentSpeech partial 1480100645859 T9 hi i am sara what is your " stroke="sp1:T9" /><event message="vrAgentSpeech partial 1480100645859 T11 hi i am sara what is your name " stroke="sp1:T11" /><sbm:event message="vrSpoke brad user 1480100642410 hi i am sara what is your name." stroke="sp1:relax" xmlns:xml="http://www.w3.org/XML/1998/namespace" xmlns:sbm="http://ict.usc.edu" /><intonation_break start="sp1:T0" end="sp1:T3"/><intonation_tone endtone="L-H%" start="sp1:T0" end="sp1:T4"/><intonation_tone endtone="L-L%" start="sp1:T3" end="sp1:T4"/><face type="FACS" au="2" amount="5.0"/><animation stroke="sp1:T3" priority="1" name="beat_low_right_Sara"/><head type="NOD" amount="0.1" repeats="1" relax="sp1:T3" priority="1" /><face type="FACS" stroke="sp1:T5" au="101" amount="1.0" priority="1" /><face type="FACS" start="sp1:T0" end="sp1:T0" au="103" amount="0.5"/><animation start="sp1:T0" priority="1" name="beat_middle_right_Sara"/><gaze participant = "Brad" target="Front" angle="0" start="7" sbm:joint-range="EYES" xmlns:sbm="http://ict.usc.edu"/></bml></act>'
# body2 = 'vrSpeak Brad User 1466989728367 <?xml version="1.0" encoding="utf-16"?><act><participant id="Brad" role="actor" /><bml><event message="vrSpoke Brad User 6 " xmlns:sbm="​http://ict.usc.edu​" /> <animation priority="1" name="beat_low_right_sara" /></bml></act>'

body3 = 'vrWEFUI start'

headers['MESSAGE_PREFIX'] = 'vrExpress'
conn.send(body=body3, destination='/topic/DEFAULT_SCOPE', headers=headers, persistent= True)

headers['MESSAGE_PREFIX'] = 'vrSpeak'
conn.send(body=body1, destination='/topic/DEFAULT_SCOPE', headers=headers, persistent= True)