import React, { useState, useEffect } from 'react';
import type { Stream, Zone, EventItem } from '../types';
import { ZoneDrawingCanvas } from '../components/ZoneDrawingCanvas';
import { SnapshotModal } from '../components/SnapshotModal';
import { updateZones, fetchZones } from '../api/client';
import { Activity, Users, ShieldAlert, Video, Eye, Shield, RefreshCw } from 'lucide-react';

interface LiveMonitorPageProps {
  streams: Stream[];
  activeStreamId: string | null;
  onSelectStream: (id: string) => void;
  recentEvents: EventItem[];
  isAlarmActive: boolean;
}

export const LiveMonitorPage: React.FC<LiveMonitorPageProps> = ({
  streams,
  activeStreamId,
  onSelectStream,
  recentEvents,
  isAlarmActive,
}) => {
  const [currentZones, setCurrentZones] = useState<Zone[]>([]);
  const [isSavingZones, setIsSavingZones] = useState(false);
  const [modalSnapshotUrl, setModalSnapshotUrl] = useState<string | null>(null);
  const [imgKey, setImgKey] = useState(Date.now());

  const currentStream = streams.find((s) => s.id === activeStreamId) || streams[0];

  useEffect(() => {
    if (currentStream?.id) {
      fetchZones(currentStream.id)
        .then(setCurrentZones)
        .catch(console.error);
    }
  }, [currentStream?.id]);

  const handleSaveZones = async (updatedZones: Zone[]) => {
    if (!currentStream?.id) return;
    setIsSavingZones(true);
    try {
      const res = await updateZones(currentStream.id, updatedZones);
      setCurrentZones(res);
      // Force refresh MJPEG preview to show newly loaded zones
      setImgKey(Date.now());
    } catch (err) {
      console.error('Failed to save zones:', err);
      alert('Failed to save zones to backend.');
    } finally {
      setIsSavingZones(false);
    }
  };

  if (!currentStream) {
    return (
      <div className="flex flex-col items-center justify-center p-16 text-center glass-panel rounded-2xl max-w-xl mx-auto my-12">
        <Video className="w-16 h-16 text-slate-600 mb-4 animate-bounce" />
        <h3 className="text-xl font-bold text-white">No Video Streams Configured</h3>
        <p className="text-sm text-slate-400 mt-2">
          Head over to the <strong className="text-red-400">Sources</strong> tab to add a webcam, RTSP stream, or upload a recorded surveillance video.
        </p>
      </div>
    );
  }

  const mjpegUrl = `/api/streams/${currentStream.id}/mjpeg?t=${imgKey}`;

  return (
    <div className="space-y-6">
      {/* Top Stream Selector and Live Metrics HUD */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 glass-panel p-4 rounded-2xl">
        <div className="flex items-center space-x-3">
          <label className="text-xs font-mono uppercase tracking-wider text-slate-400">Active Camera:</label>
          <select
            value={currentStream.id}
            onChange={(e) => onSelectStream(e.target.value)}
            className="bg-slate-900 border border-slate-700 text-white text-sm font-semibold rounded-xl px-3.5 py-1.5 focus:outline-none focus:border-red-500"
          >
            {streams.map((s) => (
              <option key={s.id} value={s.id}>
                {s.name} ({s.source_type.toUpperCase()})
              </option>
            ))}
          </select>
          <button
            onClick={() => setImgKey(Date.now())}
            title="Refresh Video Stream"
            className="p-1.5 rounded-lg text-slate-400 hover:text-white bg-slate-800 hover:bg-slate-700 transition-colors"
          >
            <RefreshCw className="w-4 h-4" />
          </button>
        </div>

        {/* Live HUD Cards */}
        <div className="grid grid-cols-3 gap-3">
          <div className="glass-card px-4 py-2 rounded-xl flex items-center space-x-2.5">
            <Activity className="w-4 h-4 text-emerald-400" />
            <div>
              <div className="text-[10px] uppercase font-mono text-slate-400">FPS</div>
              <div className="text-sm font-bold text-white">{currentStream.fps ? currentStream.fps.toFixed(1) : '25.0'}</div>
            </div>
          </div>

          <div className="glass-card px-4 py-2 rounded-xl flex items-center space-x-2.5">
            <Users className="w-4 h-4 text-cyan-400" />
            <div>
              <div className="text-[10px] uppercase font-mono text-slate-400">Resolution</div>
              <div className="text-sm font-bold text-white">{currentStream.width}x{currentStream.height}</div>
            </div>
          </div>

          <div className={`px-4 py-2 rounded-xl flex items-center space-x-2.5 transition-all ${
            isAlarmActive ? 'bg-red-600/30 border border-red-500/80 animate-alarm-pulse' : 'glass-card'
          }`}>
            <ShieldAlert className={`w-4 h-4 ${isAlarmActive ? 'text-red-400 animate-bounce' : 'text-slate-400'}`} />
            <div>
              <div className="text-[10px] uppercase font-mono text-slate-400">Status</div>
              <div className={`text-sm font-bold ${isAlarmActive ? 'text-red-400' : 'text-emerald-400'}`}>
                {isAlarmActive ? 'ALARM ACTIVE' : 'MONITORING'}
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Main Video & Interactive Zone Canvas Area */}
      <ZoneDrawingCanvas
        videoUrl={mjpegUrl}
        zones={currentZones}
        onSaveZones={handleSaveZones}
        isSaving={isSavingZones}
        isAlarmActive={isAlarmActive}
      />

      {/* Live Event Ticker below monitor */}
      <div className="glass-panel p-4 rounded-2xl border border-slate-800">
        <div className="flex items-center justify-between pb-3 border-b border-slate-800">
          <div className="flex items-center space-x-2">
            <Shield className="w-4 h-4 text-red-500" />
            <h4 className="text-xs font-bold text-white uppercase tracking-wider">Live Detection Feed (Recent Activity)</h4>
          </div>
          <span className="text-xs font-mono text-slate-400">Total: {recentEvents.length} events</span>
        </div>

        <div className="mt-3 grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 gap-3">
          {recentEvents.slice(0, 4).map((ev) => (
            <div
              key={ev.id}
              onClick={() => ev.snapshot_url && setModalSnapshotUrl(ev.snapshot_url)}
              className="p-3 rounded-xl bg-slate-900/80 border border-slate-800 hover:border-slate-700 transition-all cursor-pointer group flex flex-col justify-between"
            >
              <div className="flex items-start justify-between">
                <div>
                  <span className={`text-[10px] font-bold px-2 py-0.5 rounded font-mono ${
                    ev.event_type === 'ENTER' ? 'bg-red-500/20 text-red-400 border border-red-500/30' :
                    ev.event_type === 'LOITER' ? 'bg-amber-500/20 text-amber-400 border border-amber-500/30' :
                    'bg-blue-500/20 text-blue-400 border border-blue-500/30'
                  }`}>
                    {ev.event_type}
                  </span>
                  <div className="text-xs font-bold text-white mt-1.5">{ev.zone_name || 'Protected Zone'}</div>
                </div>
                <span className="text-[10px] font-mono text-slate-500">#{ev.track_id}</span>
              </div>

              <div className="mt-3 flex items-center justify-between text-[11px] text-slate-400 font-mono">
                <span>{new Date(ev.timestamp).toLocaleTimeString()}</span>
                {ev.snapshot_url && (
                  <span className="text-red-400 group-hover:underline flex items-center space-x-1">
                    <Eye className="w-3 h-3" />
                    <span>Snapshot</span>
                  </span>
                )}
              </div>
            </div>
          ))}
          {recentEvents.length === 0 && (
            <div className="col-span-4 py-4 text-center text-xs text-slate-500 font-mono">
              No intrusions detected yet. Protected zone is clear.
            </div>
          )}
        </div>
      </div>

      {/* Snapshot Modal */}
      <SnapshotModal
        imageUrl={modalSnapshotUrl}
        onClose={() => setModalSnapshotUrl(null)}
      />
    </div>
  );
};
