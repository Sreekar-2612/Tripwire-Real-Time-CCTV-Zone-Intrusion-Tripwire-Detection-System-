import React, { useState, useEffect } from 'react';
import { BarChart3, ShieldCheck, ShieldAlert, Clock, Activity, Video, TrendingUp, RefreshCw } from 'lucide-react';

interface AnalyticsSummary {
  total_all_time: number;
  total_24h: number;
  total_7d: number;
  active_streams_count: number;
  armed_zones_count: number;
  event_type_distribution: Record<string, number>;
  zone_distribution: Array<{ zone_name: string; count: number }>;
  hourly_distribution: number[];
  system_status: string;
}

export const AnalyticsPage: React.FC = () => {
  const [data, setData] = useState<AnalyticsSummary | null>(null);
  const [isLoading, setIsLoading] = useState(false);

  const fetchAnalytics = async () => {
    setIsLoading(true);
    try {
      const res = await fetch('/api/analytics/summary');
      if (res.ok) {
        const json = await res.json();
        setData(json);
      }
    } catch (e) {
      console.error('Failed to load analytics:', e);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    fetchAnalytics();
  }, []);

  if (!data) {
    return <div className="p-8 text-center text-slate-500 font-mono">Loading Security Analytics...</div>;
  }

  const maxHourCount = Math.max(1, ...data.hourly_distribution);

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="glass-panel p-6 rounded-2xl border border-slate-800 flex items-center justify-between">
        <div className="flex items-center space-x-3">
          <div className="p-2.5 rounded-xl bg-red-600/10 border border-red-500/30 text-red-500">
            <BarChart3 className="w-6 h-6" />
          </div>
          <div>
            <h2 className="text-xl font-bold text-white tracking-wide">Intrusion Intelligence & Analytics</h2>
            <p className="text-xs text-slate-400 mt-0.5">
              Comprehensive telemetry, 24-hour activity heatmaps, and zone breach risk metrics.
            </p>
          </div>
        </div>

        <button
          onClick={fetchAnalytics}
          className="flex items-center space-x-1.5 px-3 py-1.5 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-300 hover:text-white text-xs font-mono transition-colors cursor-pointer"
        >
          <RefreshCw className={`w-3.5 h-3.5 ${isLoading ? 'animate-spin' : ''}`} />
          <span>Refresh</span>
        </button>
      </div>

      {/* KPI Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <div className="glass-card p-5 rounded-2xl flex items-center justify-between">
          <div>
            <div className="text-[11px] font-mono uppercase text-slate-400">Total Intrusions (24h)</div>
            <div className="text-2xl font-bold text-white mt-1">{data.total_24h}</div>
            <div className="text-[10px] text-slate-500 mt-1 font-mono">{data.total_7d} recorded past 7 days</div>
          </div>
          <div className="p-3 rounded-xl bg-red-500/10 border border-red-500/30 text-red-400">
            <ShieldAlert className="w-6 h-6" />
          </div>
        </div>

        <div className="glass-card p-5 rounded-2xl flex items-center justify-between">
          <div>
            <div className="text-[11px] font-mono uppercase text-slate-400">System Security Posture</div>
            <div className={`text-xl font-bold mt-1 ${data.total_24h === 0 ? 'text-emerald-400' : 'text-amber-400'}`}>
              {data.total_24h === 0 ? 'SECURE' : 'BREACH DETECTED'}
            </div>
            <div className="text-[10px] text-slate-500 mt-1 font-mono">Continuous AI monitoring</div>
          </div>
          <div className={`p-3 rounded-xl border ${
            data.total_24h === 0 ? 'bg-emerald-500/10 border-emerald-500/30 text-emerald-400' : 'bg-amber-500/10 border-amber-500/30 text-amber-400'
          }`}>
            <ShieldCheck className="w-6 h-6" />
          </div>
        </div>

        <div className="glass-card p-5 rounded-2xl flex items-center justify-between">
          <div>
            <div className="text-[11px] font-mono uppercase text-slate-400">Active Cameras</div>
            <div className="text-2xl font-bold text-white mt-1">{data.active_streams_count}</div>
            <div className="text-[10px] text-slate-500 mt-1 font-mono">RTSP / USB Webcams</div>
          </div>
          <div className="p-3 rounded-xl bg-cyan-500/10 border border-cyan-500/30 text-cyan-400">
            <Video className="w-6 h-6" />
          </div>
        </div>

        <div className="glass-card p-5 rounded-2xl flex items-center justify-between">
          <div>
            <div className="text-[11px] font-mono uppercase text-slate-400">Actively Armed Zones</div>
            <div className="text-2xl font-bold text-white mt-1">{data.armed_zones_count}</div>
            <div className="text-[10px] text-slate-500 mt-1 font-mono">Polygon & Tripwire guards</div>
          </div>
          <div className="p-3 rounded-xl bg-purple-500/10 border border-purple-500/30 text-purple-400">
            <TrendingUp className="w-6 h-6" />
          </div>
        </div>
      </div>

      {/* 24-Hour Activity Histogram Chart */}
      <div className="glass-panel p-6 rounded-2xl border border-slate-800 space-y-4">
        <div className="flex items-center space-x-2 text-white border-b border-slate-800 pb-3">
          <Clock className="w-4 h-4 text-cyan-400" />
          <h3 className="text-sm font-bold uppercase tracking-wider">24-Hour Intrusion Distribution (Hourly Histogram)</h3>
        </div>

        <div className="h-44 flex items-end justify-between gap-1 pt-6 pb-2">
          {data.hourly_distribution.map((count, hour) => {
            const heightPercent = Math.max(8, (count / maxHourCount) * 100);
            const isPeak = count > 0 && count === maxHourCount;
            return (
              <div key={hour} className="flex-1 flex flex-col items-center group relative">
                {/* Tooltip on hover */}
                <div className="opacity-0 group-hover:opacity-100 absolute -top-8 px-2 py-1 rounded bg-slate-800 border border-slate-700 text-[10px] font-mono text-white transition-opacity pointer-events-none whitespace-nowrap z-20">
                  {hour}:00 - {count} events
                </div>

                <div
                  className={`w-full rounded-t transition-all duration-300 ${
                    count === 0
                      ? 'bg-slate-800/40 hover:bg-slate-700/60'
                      : isPeak
                      ? 'bg-red-500 shadow-md shadow-red-950/80 animate-pulse'
                      : 'bg-red-500/60 hover:bg-red-400'
                  }`}
                  style={{ height: `${heightPercent}%` }}
                />
                <span className="text-[9px] font-mono text-slate-500 mt-2">
                  {hour % 3 === 0 ? `${hour}h` : ''}
                </span>
              </div>
            );
          })}
        </div>
      </div>

      {/* Breakdown by Zone & Event Type */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Most Breached Zones */}
        <div className="glass-panel p-6 rounded-2xl border border-slate-800 space-y-4">
          <h3 className="text-sm font-bold text-white uppercase tracking-wider flex items-center space-x-2 border-b border-slate-800 pb-3">
            <Activity className="w-4 h-4 text-red-400" />
            <span>Highest Risk Perimeter Zones (Past 7 Days)</span>
          </h3>

          <div className="space-y-3">
            {data.zone_distribution.length === 0 ? (
              <div className="py-8 text-center text-xs text-slate-500 font-mono">No zone breaches recorded yet.</div>
            ) : (
              data.zone_distribution.map((z, idx) => (
                <div key={idx} className="flex items-center justify-between p-3 rounded-xl bg-slate-900/60 border border-slate-800 font-mono text-xs">
                  <div className="flex items-center space-x-2">
                    <span className="w-5 h-5 rounded-full bg-red-500/20 text-red-400 flex items-center justify-center font-bold text-[10px]">
                      {idx + 1}
                    </span>
                    <span className="text-white font-semibold">{z.zone_name}</span>
                  </div>
                  <span className="px-2.5 py-0.5 rounded-full bg-red-600/20 text-red-400 border border-red-500/30 font-bold">
                    {z.count} breaches
                  </span>
                </div>
              ))
            )}
          </div>
        </div>

        {/* Event Type Proportions */}
        <div className="glass-panel p-6 rounded-2xl border border-slate-800 space-y-4">
          <h3 className="text-sm font-bold text-white uppercase tracking-wider flex items-center space-x-2 border-b border-slate-800 pb-3">
            <BarChart3 className="w-4 h-4 text-purple-400" />
            <span>Security Classification Breakdown</span>
          </h3>

          <div className="space-y-3 font-mono text-xs">
            {Object.entries(data.event_type_distribution).length === 0 ? (
              <div className="py-8 text-center text-xs text-slate-500">No events recorded yet.</div>
            ) : (
              Object.entries(data.event_type_distribution).map(([type, count]) => (
                <div key={type} className="flex items-center justify-between p-3 rounded-xl bg-slate-900/60 border border-slate-800">
                  <span className={`px-2 py-0.5 rounded text-[10px] font-bold ${
                    type === 'ENTER' ? 'bg-red-500/20 text-red-400 border border-red-500/30' :
                    type === 'LOITER' ? 'bg-amber-500/20 text-amber-400 border border-amber-500/30' :
                    type.startsWith('CROSS') ? 'bg-purple-500/20 text-purple-400 border border-purple-500/30' :
                    'bg-blue-500/20 text-blue-400 border border-blue-500/30'
                  }`}>
                    {type}
                  </span>
                  <span className="text-white font-bold">{count} occurrences</span>
                </div>
              ))
            )}
          </div>
        </div>
      </div>
    </div>
  );
};
