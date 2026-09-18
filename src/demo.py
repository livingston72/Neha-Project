"""Local WAV-upload demo using the same extractor and saved model as training."""
from __future__ import annotations

import json
import tempfile
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from .common import log_error
from .training import load_bundle, predict_audio

HTML = """<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Clean vs Block · LTAS demo</title>
<style>body{font:17px system-ui;background:#f3f6fa;color:#162c42;margin:0;padding:7vh 20px}
main{max-width:700px;margin:auto;background:white;padding:36px;border-radius:18px;box-shadow:0 8px 40px #152b4210}
h1{font-size:30px;margin-top:0}p{line-height:1.6;color:#526273}input,button{font:inherit;margin:14px 0}
button{background:#185eac;color:white;border:0;border-radius:8px;padding:12px 22px;cursor:pointer}
button:disabled{opacity:.5}audio{width:100%;margin-top:15px}#result{white-space:pre-wrap;line-height:1.7}
.tag{font-size:13px;letter-spacing:1px;color:#185eac}small{color:#617488}</style>
<main><p class="tag">RESEARCH REPRODUCTION · LOCAL DEMO</p><h1>Clean or Block?</h1>
<p>Choose a WAV clip. The saved model uses the same 8 kHz preprocessing and 1,069 CQT-LTAS features as training.</p>
<label for="audio">WAV file</label><br><input id="audio" type="file" accept=".wav,audio/wav">
<audio id="player" controls hidden></audio><br><button id="go" disabled>Analyze audio</button>
<div id="result" role="status" aria-live="polite">Select a file to begin.</div>
<p><small>Model probability is an estimate, not a clinical diagnosis. This binary model does not identify the other stutter categories.</small></p></main>
<script>
const input=document.querySelector('#audio'),go=document.querySelector('#go'),result=document.querySelector('#result'),player=document.querySelector('#player');
let url;
input.onchange=()=>{go.disabled=!input.files.length;if(url)URL.revokeObjectURL(url);if(input.files.length){url=URL.createObjectURL(input.files[0]);player.src=url;player.hidden=false;result.textContent='Ready to analyze.'}};
go.onclick=async()=>{go.disabled=true;result.textContent='Extracting features and predicting…';try{
 const response=await fetch('/predict',{method:'POST',headers:{'Content-Type':'audio/wav'},body:input.files[0]});
 const data=await response.json();if(!response.ok)throw Error(data.error);
 result.textContent=`Prediction: ${data.prediction}\\nProbability for predicted class: ${(100*data.prediction_probability).toFixed(2)}%\\nBlock probability: ${(100*data.target_probability).toFixed(2)}%\\nDuration: ${data.audio.original_duration.toFixed(3)} seconds\\nOriginal sample rate: ${data.audio.original_sample_rate} Hz\\nAnalysis sample rate: ${data.audio.resampled_sample_rate} Hz\\nFeature dimension: ${data.feature_dimension}\\nProbability method: ${data.probability_method}`;
 }catch(error){result.textContent='Could not analyze: '+error.message}finally{go.disabled=false}};
</script></html>"""


def make_server(root, model_path, port=8501):
    model_path=Path(model_path)
    bundle=load_bundle(model_path)
    if bundle["stutter_type"] != "block" or bundle["feature_type"] != "CQT-LTAS":
        raise ValueError("This demo requires a Clean vs Block CQT-LTAS model")

    class Handler(BaseHTTPRequestHandler):
        def respond(self, code, body, content_type):
            encoded=body.encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type",content_type)
            self.send_header("Content-Length",str(len(encoded)))
            self.send_header("X-Content-Type-Options","nosniff")
            self.end_headers();self.wfile.write(encoded)

        def do_GET(self):
            self.respond(200,HTML,"text/html; charset=utf-8") if self.path=="/" else self.respond(404,"Not found","text/plain")

        def do_POST(self):
            if self.path != "/predict":
                return self.respond(404,json.dumps({"error":"Not found"}),"application/json")
            try:
                length=int(self.headers.get("Content-Length","0"))
                if length <= 0 or length > 25*1024*1024:
                    raise ValueError("Select a WAV file up to 25 MB")
                raw=self.rfile.read(length)
                if len(raw)!=length:
                    raise ValueError("Incomplete upload")
                # No user filename is used as a filesystem path.
                with tempfile.TemporaryDirectory(prefix="ltas_upload_") as directory:
                    path=Path(directory)/"uploaded.wav";path.write_bytes(raw)
                    result=predict_audio(model_path,path)
                self.respond(200,json.dumps(result,allow_nan=False),"application/json")
            except (ValueError,RuntimeError,OSError) as exc:
                log_error(root,"demo_upload",getattr(exc,"kind","demo_error"),str(exc))
                self.respond(400,json.dumps({"error":str(exc)}),"application/json")

    return HTTPServer(("127.0.0.1",port),Handler)


def serve(root,model_path,port=8501):
    with make_server(root,model_path,port) as server:
        print(f"Open http://127.0.0.1:{server.server_port} in your browser. Ctrl+C stops the demo.",flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
