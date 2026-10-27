import React from 'react';
import { ShieldAlert, Video, ListFilter, Sliders, Volume2, VolumeX, Radio } from 'lucide-react';

interface NavbarProps {
  currentTab: 'monitor' | 'sources' | 'events' | 'analytics' | 'settings';
  onSelectTab: (tab: 'monitor' | 'sources' | 'events' | 'analytics' | 'settings') => void;
  audioEnabled: boolean;
  onToggleAudio: () => void;
  wsConnected: boolean;
  activeAlarmsCount: number;
}

export const Navbar: React.FC<NavbarProps> = ({
  currentTab,
  onSelectTab,
  audioEnabled,
  onToggleAudio,
  wsConnected,
  activeAlarmsCount,
}) => {
  return (
    <header className="sticky top-0 z-40 w-full glass-panel border-b border-slate-800 px-4 py-3">
      <div className="max-w-7xl mx-auto flex items-center justify-between">
        {/* Brand / Logo */}
        <div className="flex items-center space-x-3 cursor-pointer" onClick={() => onSelectTab('monitor')}>
          <div className="relative p-2 rounded-xl bg-red-500/10 border border-red-500/30 text-red-500 flex items-center justify-center shadow-lg shadow-red-950/40">
            <ShieldAlert className="w-6 h-6 animate-pulse" />
            <span className="absolute -top-1 -right-1 flex h-3 w-3">
              <span className={`animate-ping absolute inline-flex h-full w-full rounded-full ${wsConnected ? 'bg-emerald-400 opacity-75' : 'bg-red-400 opacity-75'}`}></span>
              <span className={`relative inline-flex rounded-full h-3 w-3 ${wsConnected ? 'bg-emerald-500' : 'bg-red-500'}`}></span>
            </span>
          </div>
          <div>
            <div className="flex items-center space-x-2">
              <span className="text-xl font-bold tracking-wider text-white">TRIPWIRE</span>
              <span className="text-xs px-2 py-0.5 rounded bg-red-600/20 text-red-400 border border-red-500/30 font-mono">
                CCTV IDS
              </span>
            </div>
            <p className="text-xs text-slate-400 font-mono">Zone-Intrusion Detection System</p>
          </div>
        </div>

        {/* Navigation Tabs */}
        <nav className="flex items-center space-x-1 sm:space-x-2 bg-slate-900/80 p-1.5 rounded-xl border border-slate-800">
          <button
            onClick={() => onSelectTab('monitor')}
            className={`flex items-center space-x-2 px-3.5 py-1.5 rounded-lg text-sm font-medium transition-all ${
              currentTab === 'monitor'
                ? 'bg-red-600 text-white shadow-md shadow-red-900/40'
                : 'text-slate-400 hover:text-white hover:bg-slate-800/60'
            }`}
          >
            <Radio className="w-4 h-4" />
            <span>Live Monitor</span>
            {activeAlarmsCount > 0 && (
              <span className="ml-1.5 px-1.5 py-0.2 rounded-full text-xs bg-white text-red-600 font-bold animate-bounce">
                {activeAlarmsCount}
              </span>
            )}
          </button>

          <button
            onClick={() => onSelectTab('sources')}
            className={`flex items-center space-x-2 px-3.5 py-1.5 rounded-lg text-sm font-medium transition-all ${
              currentTab === 'sources'
                ? 'bg-red-600 text-white shadow-md shadow-red-900/40'
                : 'text-slate-400 hover:text-white hover:bg-slate-800/60'
            }`}
          >
            <Video className="w-4 h-4" />
            <span>Sources</span>
          </button>

          <button
            onClick={() => onSelectTab('events')}
            className={`flex items-center space-x-2 px-3.5 py-1.5 rounded-lg text-sm font-medium transition-all ${
              currentTab === 'events'
                ? 'bg-red-600 text-white shadow-md shadow-red-900/40'
                : 'text-slate-400 hover:text-white hover:bg-slate-800/60'
            }`}
          >
            <ListFilter className="w-4 h-4" />
            <span>Events</span>
          </button>

          <button
            onClick={() => onSelectTab('analytics')}
            className={`flex items-center space-x-2 px-3.5 py-1.5 rounded-lg text-sm font-medium transition-all ${
              currentTab === 'analytics'
                ? 'bg-red-600 text-white shadow-md shadow-red-900/40'
                : 'text-slate-400 hover:text-white hover:bg-slate-800/60'
            }`}
          >
            <Sliders className="w-4 h-4" />
            <span>Analytics</span>
          </button>

          <button
            onClick={() => onSelectTab('settings')}
            className={`flex items-center space-x-2 px-3.5 py-1.5 rounded-lg text-sm font-medium transition-all ${
              currentTab === 'settings'
                ? 'bg-red-600 text-white shadow-md shadow-red-900/40'
                : 'text-slate-400 hover:text-white hover:bg-slate-800/60'
            }`}
          >
            <Sliders className="w-4 h-4" />
            <span>Settings</span>
          </button>
        </nav>

        {/* Status Actions */}
        <div className="flex items-center space-x-3">
          {/* Audio siren toggle */}
          <button
            onClick={onToggleAudio}
            title={audioEnabled ? 'Siren Audio On' : 'Siren Audio Muted'}
            className={`p-2 rounded-lg border transition-all ${
              audioEnabled
                ? 'bg-emerald-500/10 border-emerald-500/40 text-emerald-400 hover:bg-emerald-500/20'
                : 'bg-slate-800/60 border-slate-700 text-slate-500 hover:text-slate-300'
            }`}
          >
            {audioEnabled ? <Volume2 className="w-4 h-4" /> : <VolumeX className="w-4 h-4" />}
          </button>

          {/* Connection status badge */}
          <div className="hidden sm:flex items-center space-x-2 px-3 py-1.5 rounded-lg bg-slate-900/60 border border-slate-800 text-xs font-mono">
            <span className={`w-2 h-2 rounded-full ${wsConnected ? 'bg-emerald-400' : 'bg-rose-500'}`} />
            <span className={wsConnected ? 'text-emerald-400' : 'text-rose-400'}>
              {wsConnected ? 'SYSTEM ARMED' : 'RECONNECTING'}
            </span>
          </div>
        </div>
      </div>
    </header>
  );
};
