import React, { useState, useEffect, useCallback } from 'react';
import type { Stream, EventItem, WSAlertMessage } from './types';
import { fetchStreams, fetchEvents } from './api/client';
import { useAlertWebSocket } from './hooks/useAlertWebSocket';
import { useAlertAudio } from './hooks/useAlertAudio';
import { Navbar } from './components/Navbar';
import { AlarmToast } from './components/AlarmToast';
import { LiveMonitorPage } from './pages/LiveMonitorPage';
import { SourcesPage } from './pages/SourcesPage';
import { EventsPage } from './pages/EventsPage';
import { AnalyticsPage } from './pages/AnalyticsPage';
import { SettingsPage } from './pages/SettingsPage';
import { SnapshotModal } from './components/SnapshotModal';

export const App: React.FC = () => {
  const [currentTab, setCurrentTab] = useState<'monitor' | 'sources' | 'events' | 'analytics' | 'settings'>('monitor');
  const [streams, setStreams] = useState<Stream[]>([]);
  const [activeStreamId, setActiveStreamId] = useState<string | null>(null);
  const [recentEvents, setRecentEvents] = useState<EventItem[]>([]);
  const [audioEnabled, setAudioEnabled] = useState(true);
  const [isAlarmActive, setIsAlarmActive] = useState(false);
  const [currentToastAlert, setCurrentToastAlert] = useState<WSAlertMessage | null>(null);
  const [previewSnapshotUrl, setPreviewSnapshotUrl] = useState<string | null>(null);

  const { playAlarmSound } = useAlertAudio();

  // Load streams
  const loadStreams = useCallback(async () => {
    try {
      const data = await fetchStreams();
      setStreams(data);
      if (data.length > 0 && !activeStreamId) {
        setActiveStreamId(data[0].id);
      }
    } catch (err) {
      console.error('Failed to load streams:', err);
    }
  }, [activeStreamId]);

  // Load events
  const loadRecentEvents = useCallback(async () => {
    try {
      const data = await fetchEvents({ limit: 20 });
      setRecentEvents(data);
    } catch (err) {
      console.error('Failed to load recent events:', err);
    }
  }, []);

  useEffect(() => {
    loadStreams();
    loadRecentEvents();
  }, [loadStreams, loadRecentEvents]);

  // WebSocket Live Alarm Listener
  const handleLiveAlert = useCallback((msg: WSAlertMessage) => {
    if (msg.event_type !== 'EXIT') {
      setIsAlarmActive(true);
      setCurrentToastAlert(msg);
      playAlarmSound(audioEnabled);

      // Flash alarm state for 4 seconds
      setTimeout(() => {
        setIsAlarmActive(false);
      }, 4000);
    }

    // Refresh recent events list
    loadRecentEvents();
  }, [audioEnabled, playAlarmSound, loadRecentEvents]);

  const { isConnected: wsConnected } = useAlertWebSocket(handleLiveAlert);

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col">
      {/* Top Navigation */}
      <Navbar
        currentTab={currentTab}
        onSelectTab={setCurrentTab}
        audioEnabled={audioEnabled}
        onToggleAudio={() => setAudioEnabled(!audioEnabled)}
        wsConnected={wsConnected}
        activeAlarmsCount={isAlarmActive ? 1 : 0}
      />

      {/* Main Page Content */}
      <main className="flex-1 max-w-7xl w-full mx-auto p-4 sm:p-6">
        {currentTab === 'monitor' && (
          <LiveMonitorPage
            streams={streams}
            activeStreamId={activeStreamId}
            onSelectStream={setActiveStreamId}
            recentEvents={recentEvents}
            isAlarmActive={isAlarmActive}
          />
        )}

        {currentTab === 'sources' && (
          <SourcesPage
            streams={streams}
            onRefreshStreams={loadStreams}
            onSelectStream={(id) => {
              setActiveStreamId(id);
              setCurrentTab('monitor');
            }}
          />
        )}

        {currentTab === 'events' && (
          <EventsPage streams={streams} />
        )}

        {currentTab === 'analytics' && (
          <AnalyticsPage />
        )}

        {currentTab === 'settings' && (
          <SettingsPage />
        )}
      </main>

      {/* Floating Alarm Banner */}
      <AlarmToast
        alert={currentToastAlert}
        onDismiss={() => setCurrentToastAlert(null)}
        onViewSnapshot={(url) => setPreviewSnapshotUrl(url)}
      />

      {/* Modal for snapshot preview */}
      <SnapshotModal
        imageUrl={previewSnapshotUrl}
        onClose={() => setPreviewSnapshotUrl(null)}
      />
    </div>
  );
};

export default App;
