import React from 'react';
import { X, Download, ShieldAlert } from 'lucide-react';

interface SnapshotModalProps {
  imageUrl: string | null;
  onClose: () => void;
}

export const SnapshotModal: React.FC<SnapshotModalProps> = ({ imageUrl, onClose }) => {
  if (!imageUrl) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/85 backdrop-blur-md animate-in fade-in duration-200">
      <div className="relative max-w-4xl w-full bg-slate-900 border border-slate-700 rounded-2xl shadow-2xl overflow-hidden">
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-slate-800 bg-slate-950/60">
          <div className="flex items-center space-x-2.5 text-red-400">
            <ShieldAlert className="w-5 h-5" />
            <h3 className="text-base font-bold text-white tracking-wide">Intrusion Evidence Snapshot</h3>
          </div>
          <div className="flex items-center space-x-2">
            <a
              href={imageUrl}
              download="intrusion_evidence.jpg"
              className="flex items-center space-x-1.5 px-3 py-1.5 text-xs font-medium text-slate-300 hover:text-white bg-slate-800 hover:bg-slate-700 rounded-lg transition-colors border border-slate-700"
            >
              <Download className="w-3.5 h-3.5" />
              <span>Download</span>
            </a>
            <button
              onClick={onClose}
              className="p-1.5 text-slate-400 hover:text-white rounded-lg hover:bg-slate-800 transition-colors"
            >
              <X className="w-5 h-5" />
            </button>
          </div>
        </div>

        {/* Image body */}
        <div className="p-4 bg-black flex items-center justify-center max-h-[75vh] overflow-hidden">
          <img
            src={imageUrl}
            alt="Intrusion Event"
            className="max-h-[70vh] w-auto object-contain rounded-lg shadow-lg border border-slate-800"
          />
        </div>
      </div>
    </div>
  );
};
