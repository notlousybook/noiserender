# noiserender ✨

cool little app that turns your videos into noise art (bad apple style!!)
it uses multiprocessing so its pretty fast ngl

## what it does
- converts videos into static/scrolling noise visuals
- keeps the audio synced up nicely
- uses all your cpu cores cuz why not
- has a thumbnail generator too if u want cool screenshots

## what u need
- python (3.x should work)
- ffmpeg (super important!!)
- some video files to mess around with

## setup (easy mode)

### 1. get ffmpeg
**windows:** open cmd/powershell and run:
```
winget install ffmpeg
```

**linux:** u probably know this already but:
```bash
sudo apt install ffmpeg
```

**mac:**
```bash
brew install ffmpeg
```

### 2. install python stuff
```bash
pip install pillow opencv-python numpy
```

### 3. run it
```bash
python noiserenderer.py
```

### 4. use it
1. click browse and pick a video
2. adjust settings if u want (or dont, defaults r fine)
3. hit "launch render" and wait
4. dont panic at the stitching part, it takes like 2 mins but its working i promise

## features n stuff

| setting | what it does |
|---------|--------------|
| output width | how wide u want the video (default 1920) |
| speed factor | 1x/2x/4x makes it more retro looking |
| chunks | more chunks = faster but more files |
| invert logic | swaps the noise around |
| noise type | random noise or scrolling noise |
| scroll direction | up/down/left/right (only for scrolling noise) |

## thumbnail generator
theres a button for making thumbnails too! u can:
- scrub through the video
- blend with original video if u want
- randomize the noise seed
- saves under 2mb automatically

## tips
- more chunks = faster render but u need more cpu cores
- 4x speed factor looks the most retro/vibey
- if it crashes, check that ffmpeg is actually installed
- the preview shows what ur gonna get

made by lousybook01 cuz yes 🎮
