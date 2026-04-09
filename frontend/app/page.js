import { FcAddImage } from "react-icons/fc";
import { FaCamera } from "react-icons/fa";
export default function Home() {
  return (
    <div className="flex-1 flex flex-col bg-slate-50">
      {/* Header */}
      <header className="bg-cyan-700  shadow-lg py-6 px-6 md:px-12 flex items-center justify-between text-white sticky top-0 z-50">
        <div className="flex items-center gap-3">
          <FaCamera className="text-4xl"/>
          <h1 className="text-3xl md:text-4xl font-extrabold tracking-tight">JustOCR</h1>
        </div>

      </header>

      {/* Main Content */}
      <main className="bg-linear-to-br from-teal-700 to-cyan-700 flex-1 flex flex-col items-center justify-center w-full py-16 px-4 md:px-8">
        <div className="max-w-5xl w-full text-center mb-12">
          <h2 className="text-5xl md:text-6xl font-extrabold text-transparent bg-clip-text bg-white mb-6 pb-2">
            No ads. No fees. Just OCR.
          </h2>
          <p className="text-xl md:text-3xl font-serif text-slate-100 max-w-3xl mx-auto leading-relaxed font-medium">
            We use our in-house AI model to scan your image.<br/> So privacy is guarenteed!
          </p>
        </div>

        {/* Upload Section */}
        <div className="w-full max-w-3xl bg-white rounded-4xl shadow-2xl overflow-hidden border-4 border-emerald-400 shadow-black">
          <div className="p-10 md:p-20 flex flex-col items-center justify-center border-4 border-dashed border-stone-400 rounded-3xl m-4 md:m-8 bg-blue-50/50 cursor-pointer group">
            <div className="p-6 bg-white rounded-full shadow-md mb-8 group-hover:scale-110 transition-transform duration-300">
               <FcAddImage className="text-teal-400 w-30 h-30"/> 
            </div>
            <h3 className="text-3xl md:text-4xl font-bold text-slate-800 mb-4 text-center">Upload your image</h3>
            <p className="text-xl md:text-2xl text-slate-500 mb-10 text-center max-w-lg">
              Drag and drop your image here, or click to browse from your device.
            </p>
            <label className="bg-teal-600 hover:bg-teal-800 active:bg-green-500 text-white text-2xl font-bold py-5 px-12 rounded-full shadow-lg hover:shadow-2xl transition-all cursor-pointer transform hover:-translate-y-1 inline-block">
              Select Image
              <input type="file" className="hidden" accept="image/*" />
            </label>
            <p className="mt-8 text-lg text-slate-400 font-medium">Supports JPG, PNG, WEBP (Max 10MB)</p>
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="bg-slate-900 text-slate-300 py-12 px-8 text-center text-lg md:text-xl">
        <div className="max-w-4xl mx-auto flex flex-col items-center gap-6">
          <div className="flex items-center gap-2">
            <span className="text-3xl font-extrabold text-white tracking-wide">JustOCR</span>
          </div>
          <p className="text-slate-400">© {new Date().getFullYear()} JustOCR. All rights reserved.</p>
          <div className="flex flex-wrap justify-center gap-8 mt-2 opacity-80 font-medium">
            <a href="#" className="hover:text-white transition-colors">Privacy Policy</a>
            <a href="#" className="hover:text-white transition-colors">Terms of Service</a>
            <a href="#" className="hover:text-white transition-colors">Contact Support</a>
          </div>
        </div>
      </footer>
    </div>
  );
}
