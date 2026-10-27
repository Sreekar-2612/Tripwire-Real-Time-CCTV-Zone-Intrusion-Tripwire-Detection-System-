import React, { useState } from 'react';
import type { Stream } from '../types';
import { createStream, uploadVideoStream, deleteStream, startStream, stopStream } from '../api/client';
import { Video, Plus, Play, Square, Trash2, UploadCloud, Webcam, Radio, AlertCircle } from 'lucide-react';

interface SourcesPageProps {
  streams: Stream[];
  onRefreshStreams: () => void;
  onSelectStream: (id: string) => void;
}

export const SourcesPage: React.FC<SourcesPageProps> = ({
  streams,
  onRefreshStreams,
  onSelectStream,
}) => {
  const [showAddModal, setShowAddModal] = useState(false);
  const [sourceType, setSourceType] = useState<'webcam' | 'rtsp' | 'file'>('webcam');
  const [name, setName] = useState('');
  const [sourceUrl, setSourceUrl] = useState('0');
  const [loopFile, setLoopFile] = useState(true);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [errorMessage, setErrorMessage] = useState('');

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!name.trim()) {
      setErrorMessage('Please provide a camera/stream name.');
      return;
    }
    setErrorMessage('');
    setIsSubmitting(true);

    try {
      if (sourceType === 'file') {
        if (!selectedFile) {
          setErrorMessage('Please choose a recorded video file to upload.');
          setIsSubmitting(false);
          return;
        }
        await uploadVideoStream(name, loopFile, selectedFile);
      } else {
        await createStream({
          name,
          source_type: sourceType,
          source_url: sourceUrl,
          is_active: true,
          loop_file: loopFile,
        });
      }

      setShowAddModal(false);
      setName('');
      setSelectedFile(null);
      onRefreshStreams();
    } catch (err: any) {
      setErrorMessage(err.message || 'Failed to add stream.');
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleToggleStream = async (stream: Stream) => {
    try {
      if (stream.is_active) {
        await stopStream(stream.id);
      } else {
        await startStream(stream.id);
      }
      onRefreshStreams();
    } catch (err) {
      console.error('Toggle stream error:', err);
    }
  };

  const handleDelete = async (streamId: string) => {
    if (!confirm('Are you sure you want to delete this camera stream and its zones?')) return;
    try {
      await deleteStream(streamId);
      onRefreshStreams();
    } catch (err) {
      console.error('Delete stream error:', err);
    }
  };

  return (
    <div className="space-y-6">
      {/* Header and Add Button */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 glass-panel p-6 rounded-2xl">
        <div>
          <h2 className="text-xl font-bold text-white tracking-wide">Video Feed Sources</h2>
          <p className="text-xs text-slate-400 mt-1">
            Connect live CCTV cameras (RTSP, ONVIF), local USB/built-in webcams, or uploaded recorded footage.
          </p>
        </div>

        <button
          onClick={() => {
            setErrorMessage('');
            setShowAddModal(true);
          }}
          className="flex items-center space-x-2 px-4 py-2.5 rounded-xl bg-red-600 hover:bg-red-500 text-white font-bold text-xs uppercase tracking-wider shadow-lg shadow-red-950/60 transition-all cursor-pointer"
        >
          <Plus className="w-4 h-4" />
          <span>Add Video Source</span>
        </button>
      </div>

      {/* Streams Cards Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {streams.map((stream) => (
          <div
            key={stream.id}
            className="glass-panel rounded-2xl p-5 border border-slate-800 flex flex-col justify-between hover:border-slate-700 transition-all shadow-xl group"
          >
            <div>
              {/* Header with Type badge and Status */}
              <div className="flex items-center justify-between mb-3">
                <span className={`text-[10px] font-bold uppercase tracking-wider px-2.5 py-1 rounded-md font-mono flex items-center space-x-1.5 ${
                  stream.source_type === 'rtsp'
                    ? 'bg-purple-500/20 text-purple-400 border border-purple-500/30'
                    : stream.source_type === 'webcam'
                    ? 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/30'
                    : 'bg-blue-500/20 text-blue-400 border border-blue-500/30'
                }`}>
                  {stream.source_type === 'rtsp' && <Radio className="w-3 h-3" />}
                  {stream.source_type === 'webcam' && <Webcam className="w-3 h-3" />}
                  {stream.source_type === 'file' && <Video className="w-3 h-3" />}
                  <span>{stream.source_type}</span>
                </span>

                <div className="flex items-center space-x-2">
                  <span className={`w-2 h-2 rounded-full ${stream.is_active ? 'bg-emerald-400 animate-ping' : 'bg-slate-600'}`} />
                  <span className={`text-xs font-mono font-medium ${stream.is_active ? 'text-emerald-400' : 'text-slate-500'}`}>
                    {stream.is_active ? 'ONLINE' : 'STOPPED'}
                  </span>
                </div>
              </div>

              {/* Title & Details */}
              <h3 className="text-base font-bold text-white group-hover:text-red-400 transition-colors">
                {stream.name}
              </h3>
              <p className="text-xs font-mono text-slate-400 mt-1 truncate" title={stream.source_url}>
                {stream.source_url}
              </p>

              {/* Stream Specs */}
              <div className="mt-4 grid grid-cols-2 gap-2 text-xs font-mono">
                <div className="bg-slate-900/80 p-2 rounded-lg border border-slate-800/80">
                  <span className="text-slate-500 block text-[10px]">FPS RATE</span>
                  <span className="text-slate-200 font-bold">{stream.fps ? stream.fps.toFixed(1) : '25.0'} FPS</span>
                </div>
                <div className="bg-slate-900/80 p-2 rounded-lg border border-slate-800/80">
                  <span className="text-slate-500 block text-[10px]">RESOLUTION</span>
                  <span className="text-slate-200 font-bold">{stream.width} x {stream.height}</span>
                </div>
              </div>
            </div>

            {/* Actions Bar */}
            <div className="mt-5 pt-4 border-t border-slate-800/80 flex items-center justify-between">
              <button
                onClick={() => onSelectStream(stream.id)}
                className="text-xs font-bold text-slate-300 hover:text-white px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 transition-colors"
              >
                View Live
              </button>

              <div className="flex items-center space-x-2">
                <button
                  onClick={() => handleToggleStream(stream)}
                  title={stream.is_active ? 'Stop Stream' : 'Start Stream'}
                  className={`p-2 rounded-lg border transition-colors ${
                    stream.is_active
                      ? 'bg-amber-500/10 border-amber-500/30 text-amber-400 hover:bg-amber-500/20'
                      : 'bg-emerald-500/10 border-emerald-500/30 text-emerald-400 hover:bg-emerald-500/20'
                  }`}
                >
                  {stream.is_active ? <Square className="w-4 h-4" /> : <Play className="w-4 h-4" />}
                </button>

                <button
                  onClick={() => handleDelete(stream.id)}
                  title="Delete Stream"
                  className="p-2 rounded-lg bg-slate-800/60 border border-slate-700 text-slate-400 hover:text-rose-400 hover:bg-rose-500/10 hover:border-rose-500/30 transition-colors"
                >
                  <Trash2 className="w-4 h-4" />
                </button>
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Add Stream Modal */}
      {showAddModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-md animate-in fade-in duration-200">
          <div className="max-w-lg w-full bg-slate-900 border border-slate-700 rounded-2xl shadow-2xl p-6">
            <h3 className="text-lg font-bold text-white mb-4">Add Surveillance Video Source</h3>

            {errorMessage && (
              <div className="mb-4 p-3 rounded-xl bg-red-500/15 border border-red-500/40 text-red-400 text-xs flex items-center space-x-2">
                <AlertCircle className="w-4 h-4 flex-shrink-0" />
                <span>{errorMessage}</span>
              </div>
            )}

            <form onSubmit={handleSubmit} className="space-y-4">
              <div>
                <label className="block text-xs font-mono uppercase tracking-wider text-slate-400 mb-1.5">
                  Camera / Stream Name
                </label>
                <input
                  type="text"
                  required
                  placeholder="e.g. North Gate Perimeter"
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                  className="w-full bg-slate-950 border border-slate-700 rounded-xl px-3.5 py-2 text-sm text-white focus:outline-none focus:border-red-500"
                />
              </div>

              {/* Source Type Selector */}
              <div>
                <label className="block text-xs font-mono uppercase tracking-wider text-slate-400 mb-1.5">
                  Source Protocol / Media
                </label>
                <div className="grid grid-cols-3 gap-2">
                  <button
                    type="button"
                    onClick={() => {
                      setSourceType('webcam');
                      setSourceUrl('0');
                    }}
                    className={`py-2 px-3 rounded-xl border text-xs font-bold transition-all flex flex-col items-center space-y-1 ${
                      sourceType === 'webcam'
                        ? 'bg-red-600/20 border-red-500 text-red-400'
                        : 'bg-slate-950 border-slate-800 text-slate-400 hover:bg-slate-800'
                    }`}
                  >
                    <Webcam className="w-4 h-4" />
                    <span>Webcam</span>
                  </button>

                  <button
                    type="button"
                    onClick={() => {
                      setSourceType('rtsp');
                      setSourceUrl('rtsp://');
                    }}
                    className={`py-2 px-3 rounded-xl border text-xs font-bold transition-all flex flex-col items-center space-y-1 ${
                      sourceType === 'rtsp'
                        ? 'bg-red-600/20 border-red-500 text-red-400'
                        : 'bg-slate-950 border-slate-800 text-slate-400 hover:bg-slate-800'
                    }`}
                  >
                    <Radio className="w-4 h-4" />
                    <span>RTSP / IP Cam</span>
                  </button>

                  <button
                    type="button"
                    onClick={() => setSourceType('file')}
                    className={`py-2 px-3 rounded-xl border text-xs font-bold transition-all flex flex-col items-center space-y-1 ${
                      sourceType === 'file'
                        ? 'bg-red-600/20 border-red-500 text-red-400'
                        : 'bg-slate-950 border-slate-800 text-slate-400 hover:bg-slate-800'
                    }`}
                  >
                    <UploadCloud className="w-4 h-4" />
                    <span>Upload Video</span>
                  </button>
                </div>
              </div>

              {/* Source Inputs */}
              {sourceType === 'webcam' && (
                <div>
                  <label className="block text-xs font-mono uppercase tracking-wider text-slate-400 mb-1.5">
                    Webcam Device Index
                  </label>
                  <input
                    type="text"
                    required
                    value={sourceUrl}
                    onChange={(e) => setSourceUrl(e.target.value)}
                    placeholder="0 (primary built-in/USB camera)"
                    className="w-full bg-slate-950 border border-slate-700 rounded-xl px-3.5 py-2 text-sm text-white focus:outline-none focus:border-red-500 font-mono"
                  />
                  <p className="text-[11px] text-slate-500 mt-1">Default webcam is index 0. External USB webcams are usually 1 or 2.</p>
                </div>
              )}

              {sourceType === 'rtsp' && (
                <div>
                  <label className="block text-xs font-mono uppercase tracking-wider text-slate-400 mb-1.5">
                    RTSP Stream URL
                  </label>
                  <input
                    type="text"
                    required
                    value={sourceUrl}
                    onChange={(e) => setSourceUrl(e.target.value)}
                    placeholder="rtsp://admin:pass@192.168.1.100:554/h264Preview_01_main"
                    className="w-full bg-slate-950 border border-slate-700 rounded-xl px-3.5 py-2 text-sm text-white focus:outline-none focus:border-red-500 font-mono"
                  />
                </div>
              )}

              {sourceType === 'file' && (
                <div>
                  <label className="block text-xs font-mono uppercase tracking-wider text-slate-400 mb-1.5">
                    Select Recorded Footage (.mp4, .avi, .mov)
                  </label>
                  <input
                    type="file"
                    accept="video/mp4,video/avi,video/quicktime,video/mkv"
                    onChange={(e) => setSelectedFile(e.target.files?.[0] || null)}
                    className="w-full text-xs text-slate-400 file:mr-4 file:py-2 file:px-4 file:rounded-xl file:border-0 file:text-xs file:font-semibold file:bg-slate-800 file:text-white hover:file:bg-slate-700 cursor-pointer"
                  />
                  <label className="flex items-center space-x-2 mt-3 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={loopFile}
                      onChange={(e) => setLoopFile(e.target.checked)}
                      className="rounded bg-slate-950 border-slate-700 text-red-600 focus:ring-0"
                    />
                    <span className="text-xs text-slate-300">Loop playback continuously for 24/7 monitoring</span>
                  </label>
                </div>
              )}

              {/* Form Buttons */}
              <div className="pt-3 flex items-center justify-end space-x-3">
                <button
                  type="button"
                  onClick={() => setShowAddModal(false)}
                  className="px-4 py-2 rounded-xl text-xs font-medium text-slate-400 hover:text-white bg-slate-800 hover:bg-slate-700 transition-colors"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  disabled={isSubmitting}
                  className="px-5 py-2 rounded-xl text-xs font-bold text-white bg-red-600 hover:bg-red-500 shadow-lg shadow-red-950/60 transition-all cursor-pointer disabled:opacity-50"
                >
                  {isSubmitting ? 'Connecting...' : 'Add Source'}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  );
};
