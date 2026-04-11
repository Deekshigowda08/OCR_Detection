"use client";
import { useState, useEffect } from "react";
import { FcAddImage } from "react-icons/fc";
import { FaCamera, FaChartLine } from "react-icons/fa";

export default function Home() {
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [metrics, setMetrics] = useState(null);

  useEffect(() => {
    fetch("http://localhost:8000/metrics")
      .then(res => res.json())
      .then(data => {
        if (!data.error) setMetrics(data);
      })
      .catch(err => console.error("Metrics load error", err));
  }, []);

  const handleUpload = async (e) => {
    const selectedFile = e.target.files[0];
    if (!selectedFile) return;
    setFile(selectedFile);
    setLoading(true);
    setResult(null);

    const formData = new FormData();
    formData.append("image", selectedFile);

    try {
      const res = await fetch("http://localhost:8000/predict", {
        method: "POST",
        body: formData,
      });
      const data = await res.json();
      console.log(data);
      setResult(data);
    } catch (err) {
      console.error(err);
      alert("Failed to connect to backend inference server.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex-1 flex flex-col bg-slate-50 min-h-screen">
      {/* Header */}
      <header className="bg-cyan-700 shadow-lg py-6 px-6 md:px-12 flex items-center justify-between text-white sticky top-0 z-50">
        <div className="flex items-center gap-3">
          <FaCamera className="text-4xl"/>
          <h1 className="text-3xl md:text-4xl font-extrabold tracking-tight">JustOCR</h1>
        </div>
      </header>

      {/* Main Content */}
      <main className="bg-linear-to-br from-teal-700 to-cyan-700 flex-1 flex flex-col items-center w-full py-16 px-4 md:px-8">
        
        {/* Intro */}
        <div className="max-w-5xl w-full text-center mb-12">
          <h2 className="text-5xl md:text-6xl font-extrabold text-transparent bg-clip-text bg-white mb-6 pb-2">
            No ads. No fees. Just OCR.
          </h2>
          <p className="text-xl md:text-3xl font-serif text-slate-100 max-w-3xl mx-auto leading-relaxed font-medium">
            We use our industrial YOLO model to scan your serial numbers.<br/> Powered by Mac MPS.
          </p>
        </div>

        {/* Inference / Upload Flow */}
        <div className="w-full max-w-5xl bg-white rounded-4xl shadow-2xl overflow-hidden border-4 border-emerald-400 mb-16">
          <div className="p-8 md:p-12 flex flex-col lg:flex-row items-center gap-8">
            
            {/* Upload Area */}
            <div className="flex-1 w-full p-8 border-4 border-dashed border-stone-300 rounded-3xl bg-blue-50/50 flex flex-col items-center justify-center relative">
              <div className="p-4 bg-white rounded-full shadow-md mb-6">
                 <FcAddImage className="text-teal-400 w-24 h-24"/> 
              </div>
              <h3 className="text-2xl font-bold text-slate-800 mb-2">Upload Image</h3>
              <p className="text-lg text-slate-500 mb-6 text-center">JPG, PNG, WEBP Supported</p>
              
              <label className="bg-teal-600 hover:bg-teal-800 active:bg-green-500 text-white text-xl font-bold py-4 px-8 rounded-full shadow-lg cursor-pointer transition-all">
                {loading ? "Analyzing..." : "Select File"}
                <input type="file" className="hidden" accept="image/*" onChange={handleUpload} disabled={loading}/>
              </label>
              {file && !loading && <p className="mt-4 text-emerald-600 font-semibold">{file.name}</p>}
            </div>

            {/* Results Area */}
            <div className="flex-1 w-full flex flex-col items-center justify-center p-4">
              {loading ? (
                <div className="animate-pulse flex flex-col items-center">
                  <div className="w-32 h-32 bg-slate-200 rounded-full mb-4 flex items-center justify-center">
                      <FaCamera className="text-6xl text-slate-400 animate-bounce" />
                  </div>
                  <h3 className="text-2xl font-bold text-slate-400">Processing with YOLOv8...</h3>
                </div>
              ) : result && !result.error ? (
                <div className="flex flex-col items-center w-full">
                  <div className="w-full bg-slate-100 rounded-2xl p-4 shadow-inner mb-6 border-2 border-slate-200">
                    <img src={result.annotated_image} alt="Annotated Detections" className="w-full rounded-xl object-contain" style={{maxHeight: "300px"}} />
                  </div>
                  
                  <div className="bg-emerald-100 border-2 border-emerald-400 w-full rounded-2xl p-6 text-center shadow-md">
                    <p className="text-slate-500 font-bold mb-2 uppercase tracking-wider">Detected String</p>
                    {result.text ? (
                      <p className="text-4xl md:text-5xl font-mono font-extrabold text-emerald-800 tracking-widest">{result.text}</p>
                    ) : (
                      <div className="mt-2 text-amber-700 bg-amber-100 p-3 rounded-lg font-medium border border-amber-300">
                        <p>No characters detected by the model.</p>
                        <p className="text-sm opacity-80 mt-1">If using the raw yolov8n.pt weight, it doesn't know letters yet!</p>
                      </div>
                    )}
                  </div>

                  {result.confidences && result.confidences.length > 0 && (
                     <div className="w-full mt-4 bg-slate-50 border rounded-xl p-4">
                     <p className="text-xs font-bold text-slate-400 uppercase mb-2">Confidence Scores</p>
                     <div className="flex flex-wrap gap-2">
                       {result.confidences.map((c, i) => (
                         <span key={i} className="px-2 py-1 bg-teal-100 text-teal-800 rounded font-mono text-sm">
                           {result.text[i] || '?'}: {(c*100).toFixed(1)}%
                         </span>
                       ))}
                     </div>
                   </div>
                  )}
                </div>
              ) : (
                <div className="text-center opacity-50 p-12">
                  <FaCamera className="text-6xl text-slate-300 mx-auto mb-4" />
                  <p className="text-2xl font-bold text-slate-400">
                    {result?.error ? result.error : "Awaiting Image..."}
                  </p>
                </div>
              )}
            </div>
            
          </div>
        </div>

        {/* Metrics Section */}
        {metrics && (
          <div className="w-full max-w-5xl bg-white rounded-4xl shadow-2xl p-8 mb-16 border-t-8 border-cyan-800">
            <div className="flex items-center gap-4 mb-8 justify-center">
              <FaChartLine className="text-4xl text-cyan-700"/>
              <h2 className="text-4xl font-extrabold text-slate-800">Model Metrics</h2>
            </div>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
              <div className="bg-slate-50 p-4 rounded-3xl shadow-inner border border-slate-200 text-center">
                <h3 className="text-xl font-bold text-slate-600 mb-4">Training Class Loss</h3>
                <img src={metrics.loss_graph} alt="Loss Graph" className="w-full rounded-xl mix-blend-multiply" />
              </div>
              <div className="bg-slate-50 p-4 rounded-3xl shadow-inner border border-slate-200 text-center">
                <h3 className="text-xl font-bold text-slate-600 mb-4">mAP (50) Timeline</h3>
                <img src={metrics.map_graph} alt="mAP Graph" className="w-full rounded-xl mix-blend-multiply" />
              </div>
            </div>
          </div>
        )}

      </main>

      {/* Footer */}
      <footer className="bg-slate-900 text-slate-300 py-12 px-8 text-center text-lg md:text-xl mt-auto">
        <div className="max-w-4xl mx-auto flex flex-col items-center gap-6">
          <div className="flex items-center gap-2">
            <span className="text-3xl font-extrabold text-white tracking-wide">JustOCR</span>
          </div>
          <p className="text-slate-400">© {new Date().getFullYear()} JustOCR YOLO. All rights reserved.</p>
        </div>
      </footer>
    </div>
  );
}
