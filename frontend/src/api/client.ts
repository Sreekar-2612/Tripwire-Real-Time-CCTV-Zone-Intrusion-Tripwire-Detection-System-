import type { Stream, Zone, EventItem, Settings } from '../types';

const BASE_URL = '';

export async function fetchStreams(): Promise<Stream[]> {
  const res = await fetch(`${BASE_URL}/api/streams`);
  if (!res.ok) throw new Error('Failed to fetch streams');
  return res.json();
}

export async function createStream(data: {
  name: string;
  source_type: string;
  source_url: string;
  is_active?: boolean;
  loop_file?: boolean;
}): Promise<Stream> {
  const res = await fetch(`${BASE_URL}/api/streams`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data),
  });
  if (!res.ok) throw new Error('Failed to create stream');
  return res.json();
}

export async function uploadVideoStream(name: string, loop_file: boolean, file: File): Promise<Stream> {
  const formData = new FormData();
  formData.append('name', name);
  formData.append('loop_file', String(loop_file));
  formData.append('file', file);

  const res = await fetch(`${BASE_URL}/api/streams/upload`, {
    method: 'POST',
    body: formData,
  });
  if (!res.ok) throw new Error('Failed to upload video stream');
  return res.json();
}

export async function deleteStream(streamId: string): Promise<void> {
  const res = await fetch(`${BASE_URL}/api/streams/${streamId}`, {
    method: 'DELETE',
  });
  if (!res.ok) throw new Error('Failed to delete stream');
}

export async function startStream(streamId: string): Promise<void> {
  const res = await fetch(`${BASE_URL}/api/streams/${streamId}/start`, {
    method: 'POST',
  });
  if (!res.ok) throw new Error('Failed to start stream');
}

export async function stopStream(streamId: string): Promise<void> {
  const res = await fetch(`${BASE_URL}/api/streams/${streamId}/stop`, {
    method: 'POST',
  });
  if (!res.ok) throw new Error('Failed to stop stream');
}

export async function fetchZones(streamId: string): Promise<Zone[]> {
  const res = await fetch(`${BASE_URL}/api/streams/${streamId}/zones`);
  if (!res.ok) throw new Error('Failed to fetch zones');
  return res.json();
}

export async function updateZones(streamId: string, zones: Zone[]): Promise<Zone[]> {
  const res = await fetch(`${BASE_URL}/api/streams/${streamId}/zones`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(zones),
  });
  if (!res.ok) throw new Error('Failed to update zones');
  return res.json();
}

export async function fetchEvents(params?: {
  stream_id?: string;
  event_type?: string;
  limit?: number;
  offset?: number;
}): Promise<EventItem[]> {
  const url = new URL(`${window.location.origin}/api/events`);
  if (params?.stream_id) url.searchParams.append('stream_id', params.stream_id);
  if (params?.event_type) url.searchParams.append('event_type', params.event_type);
  if (params?.limit) url.searchParams.append('limit', String(params.limit));
  if (params?.offset) url.searchParams.append('offset', String(params.offset));

  const res = await fetch(url.toString());
  if (!res.ok) throw new Error('Failed to fetch events');
  return res.json();
}

export async function deleteEvent(eventId: string): Promise<void> {
  const res = await fetch(`${BASE_URL}/api/events/${eventId}`, {
    method: 'DELETE',
  });
  if (!res.ok) throw new Error('Failed to delete event');
}

export async function fetchSettings(): Promise<Settings> {
  const res = await fetch(`${BASE_URL}/api/settings`);
  if (!res.ok) throw new Error('Failed to fetch settings');
  return res.json();
}

export async function updateSettings(settings: Settings): Promise<Settings> {
  const res = await fetch(`${BASE_URL}/api/settings`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(settings),
  });
  if (!res.ok) throw new Error('Failed to update settings');
  return res.json();
}

export async function testAlertChannel(channel: 'telegram' | 'webhook' | 'email'): Promise<{ status: string; message: string }> {
  const res = await fetch(`${BASE_URL}/api/settings/test-alert`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ channel }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Failed' }));
    throw new Error(err.detail || 'Test alert failed');
  }
  return res.json();
}
