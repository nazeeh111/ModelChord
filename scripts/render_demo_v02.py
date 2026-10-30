#!/usr/bin/env python3
"""Caption actual completed UI captures; no mocked UI or synthetic score states."""
from pathlib import Path
import hashlib,json,subprocess,shutil
root=Path(__file__).resolve().parents[1];media=root/'docs/media/v02';out=media/'render-1'
if out.exists():raise SystemExit('Keep the previous render; choose a fresh output folder')
out.mkdir()
shots=[('build-result.png',8,'Selection: 4/4 required floors and planning targets. Rank 6 folded into one model.'),
       ('inference-before.png',6,'Real static-artifact inference: digit 8 receives 99.2% confidence.'),
       ('inference-after.png',6,'Two pixels changed: confidence becomes 98.9%. Scores respond to current inputs.')]
font='/System/Library/Fonts/Helvetica.ttc';clips=[];manifest=[]
for i,(file,duration,caption) in enumerate(shots):
 source=media/file;caption_path=out/f'caption-{i}.txt';caption_path.write_text(caption)
 clip=out/f'clip-{i}.mp4'
 vf=f"scale=1470:800:force_original_aspect_ratio=decrease,pad=1470:880:0:0:color=0xf4f5ef,drawtext=fontfile={font}:textfile={caption_path}:expansion=none:x=28:y=817:fontsize=22:fontcolor=0x223330,drawtext=fontfile={font}:text='Actual local UI captures - edited for time - no audio':x=28:y=851:fontsize=14:fontcolor=0x536461"
 subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-nostdin','-n','-loop','1','-i',str(source),'-t',str(duration),'-vf',vf,'-r','30','-c:v','libx264','-preset','fast','-crf','20','-pix_fmt','yuv420p',str(clip)],check=True)
 clips.append(clip);manifest.append({'source':file,'sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'duration_seconds':duration,'caption':caption})
listing=out/'clips.txt';listing.write_text(''.join("file '"+str(p)+"'\n" for p in clips))
master=media/'MODELCHORD-v0.2.0.mp4'
subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-nostdin','-n','-f','concat','-safe','0','-i',str(listing),'-c','copy','-movflags','+faststart',str(master)],check=True)
probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames','-show_entries','stream=width,height,codec_name,avg_frame_rate,nb_read_frames:format=duration','-of','json',str(master)]))
assert probe['streams'][0]['nb_read_frames']=='600' and abs(float(probe['format']['duration'])-20)<.001
subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-nostdin','-i',str(master),'-f','null','-'],check=True)
(media/'demo-manifest.json').write_text(json.dumps({'shots':manifest,'master_sha256':hashlib.sha256(master.read_bytes()).hexdigest(),'probe':probe,'capture_note':'Completed real UI states with original scores, captioned and held for reading. Captioned still captures of actual executed v0.2 states, not a continuous screen recording. Selection scores are not new-cohort confirmation scores.'},indent=2)+'\n')
print(json.dumps({'master':str(master),'probe':probe},indent=2))
