import React, { useState, useEffect } from 'react';
import type { EventItem, Stream } from '../types';
import { fetchEvents, deleteEvent } from '../api/client';
import { SnapshotModal } from '../components/SnapshotModal';
import { ClipModal } from '../components/ClipModal';
import { Download, RefreshCw, Trash2, Film, Eye } from 'lucide-react';

interface EventsPageProps {
  streams: Stream[];
}

export const EventsPage: React.FC<EventsPageProps> = ({ streams }) => {
  const [events, setEvents] = useState<EventItem[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [selectedStreamId, setSelectedStreamId] = useState<string>('');
  const [selectedEventType, setSelectedEventType] = useState<string>('');
  const [selectedSnapshot, setSelectedSnapshot] = useState<string | null>(null);
  const [selectedClip, setSelectedClip] = useState<string | null>(null);

  const loadEvents = async () => {
    setIsLoading(true);
    try {
      const data = await fetchEvents({
        stream_id: selectedStreamId || undefined,
        event_type: selectedEventType || undefined,
        limit: 100,
      });
      setEvents(data);
    } catch (err) {
      console.error('Error fetching events:', err);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    loadEvents();
  }, [selectedStreamId, selectedEventType]);

  const handleDelete = async (id: string) => {
    if (!confirm('Are you sure you want to delete this event and its evidence files?')) return;
    try {
      await deleteEvent(id);
      setEvents((prev) => prev.filter((e) => e.id !== id));
    } catch (err) {
      console.error('Delete event error:', err);
    }
  };

  const handleExportCSV = () => {
    const url = `/api/events/export/csv${selectedStreamId ? `?stream_id=${selectedStreamId}` : ''}`;
    window.open(url, '_blank');
  };

  return (
    <div className="space-y-6">
      {/* Filters & Export Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 glass-panel p-5 rounded-2xl">
        <div className="flex flex-wrap items-center gap-3">
          {/* Stream Filter */}
          <div className="flex items-center space-x-2">
            <span className="text-xs font-mono text-slate-400">CAMERA:</span>
            <select
              value={selectedStreamId}
              onChange={(e) => setSelectedStreamId(e.target.value)}
              className="bg-slate-900 border border-slate-700 text-white text-xs rounded-xl px-3 py-1.5 focus:outline-none focus:border-red-500"
            >
              <option value="">All Cameras</option>
              {streams.map((s) => (
                <option key={s.id} value={s.id}>
                  {s.name}
                </option>
              ))}
            </select>
          </div>

          {/* Event Type Filter */}
          <div className="flex items-center space-x-2">
            <span className="text-xs font-mono text-slate-400">TYPE:</span>
            <select
              value={selectedEventType}
              onChange={(e) => setSelectedEventType(e.target.value)}
              className="bg-slate-900 border border-slate-700 text-white text-xs rounded-xl px-3 py-1.5 focus:outline-none focus:border-red-500"
            >
              <option value="">All Event Types</option>
              <option value="ENTER">ENTER (Intrusion)</option>
              <option value="LOITER">LOITER (Loitering)</option>
              <option value="CROSS_IN">CROSS_IN (Tripwire)</option>
              <option value="CROSS_OUT">CROSS_OUT (Tripwire)</option>
              <option value="EXIT">EXIT</option>
            </select>
          </div>

          <button
            onClick={loadEvents}
            disabled={isLoading}
            className="p-1.5 rounded-lg text-slate-400 hover:text-white bg-slate-800 hover:bg-slate-700 transition-colors"
            title="Refresh events"
          >
            <RefreshCw className={`w-4 h-4 ${isLoading ? 'animate-spin' : ''}`} />
          </button>
        </div>

        {/* Export CSV Button */}
        <button
          onClick={handleExportCSV}
          className="flex items-center space-x-2 px-4 py-2 rounded-xl bg-slate-800 hover:bg-slate-700 text-white text-xs font-bold border border-slate-700 transition-all cursor-pointer shadow-md"
        >
          <Download className="w-3.5 h-3.5 text-red-400" />
          <span>Export Audit Log (CSV)</span>
        </button>
      </div>

      {/* Events Table */}
      <div className="glass-panel rounded-2xl overflow-hidden border border-slate-800">
        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs font-mono">
            <thead className="bg-slate-900/90 text-slate-400 uppercase tracking-wider border-b border-slate-800">
              <tr>
                <th className="py-3.5 px-4">Snapshot</th>
                <th className="py-3.5 px-4">Timestamp</th>
                <th className="py-3.5 px-4">Event</th>
                <th className="py-3.5 px-4">Zone</th>
                <th className="py-3.5 px-4">Track ID</th>
                <th className="py-3.5 px-4">Confidence</th>
                <th className="py-3.5 px-4 text-center">Clip Video</th>
                <th className="py-3.5 px-4 text-right">Actions</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/60">
              {events.map((ev) => (
                <tr key={ev.id} className="hover:bg-slate-800/40 transition-colors">
                  {/* Thumbnail */}
                  <td className="py-3 px-4">
                    {ev.snapshot_url ? (
                      <div
                        onClick={() => setSelectedSnapshot(ev.snapshot_url!)}
                        className="w-14 h-10 rounded-lg overflow-hidden bg-black border border-slate-700 cursor-pointer hover:border-red-500 transition-colors group relative"
                      >
                        <img
                          src={ev.snapshot_url}
                          alt="Snapshot"
                          className="w-full h-full object-cover group-hover:scale-110 transition-transform"
                        />
                        <div className="absolute inset-0 bg-black/30 flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity">
                          <Eye className="w-3 h-3 text-white" />
                        </div>
                      </div>
                    ) : (
                      <span className="text-slate-600">—</span>
                    )}
                  </td>

                  {/* Timestamp */}
                  <td className="py-3 px-4 text-slate-300">
                    <div>{new Date(ev.timestamp).toLocaleDateString()}</div>
                    <div className="text-[11px] text-slate-500">{new Date(ev.timestamp).toLocaleTimeString()}</div>
                  </td>

                  {/* Event Type Badge */}
                  <td className="py-3 px-4">
                    <span
                      className={`px-2 py-0.5 rounded text-[10px] font-bold ${
                        ev.event_type === 'ENTER'
                          ? 'bg-red-500/20 text-red-400 border border-red-500/30'
                          : ev.event_type === 'LOITER'
                          ? 'bg-amber-500/20 text-amber-400 border border-amber-500/30'
                          : ev.event_type === 'CROSS_IN' || ev.event_type === 'CROSS_OUT'
                          ? 'bg-purple-500/20 text-purple-400 border border-purple-500/30'
                          : 'bg-slate-700/40 text-slate-400'
                      }`}
                    >
                      {ev.event_type}
                    </span>
                  </td>

                  {/* Zone Name */}
                  <td className="py-3 px-4 font-bold text-white">
                    {ev.zone_name || 'Protected Zone'}
                  </td>

                  {/* Track ID */}
                  <td className="py-3 px-4 text-slate-400">
                    #{ev.track_id}
                  </td>

                  {/* Confidence */}
                  <td className="py-3 px-4 text-slate-300">
                    {(ev.confidence * 100).toFixed(0)}%
                  </td>

                  {/* Video Clip Play Button */}
                  <td className="py-3 px-4 text-center">
                    {ev.clip_url ? (
                      <button
                        onClick={() => setSelectedClip(ev.clip_url!)}
                        className="inline-flex items-center space-x-1 px-2.5 py-1 rounded-lg bg-red-600/20 hover:bg-red-600/30 text-red-400 border border-red-500/40 transition-colors cursor-pointer"
                      >
                        <Film className="w-3 h-3" />
                        <span>Play Clip</span>
                      </button>
                    ) : (
                      <span className="text-slate-600 text-[11px]">No Clip</span>
                    )}
                  </td>

                  {/* Actions */}
                  <td className="py-3 px-4 text-right">
                    <button
                      onClick={() => handleDelete(ev.id)}
                      className="p-1.5 text-slate-500 hover:text-rose-400 hover:bg-slate-800 rounded-lg transition-colors"
                      title="Delete event record"
                    >
                      <Trash2 className="w-3.5 h-3.5" />
                    </button>
                  </td>
                </tr>
              ))}

              {events.length === 0 && !isLoading && (
                <tr>
                  <td colSpan={8} className="py-12 text-center text-slate-500">
                    No intrusion alarm events found matching the criteria.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>

      {/* Modals */}
      <SnapshotModal
        imageUrl={selectedSnapshot}
        onClose={() => setSelectedSnapshot(null)}
      />

      <ClipModal
        clipUrl={selectedClip}
        onClose={() => setSelectedClip(null)}
      />
    </div>
  );
};
