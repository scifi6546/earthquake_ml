import obspy as ob
from obspy.clients.fdsn import Client
t = ob.UTCDateTime("2010-02-27T06:45:00.000")
client = Client("IRIS")
st = client.get_waveforms("IU", "ANMO", "00", "LHZ", t, t + 60 * 60)
print(st[0])
st.plot()  


