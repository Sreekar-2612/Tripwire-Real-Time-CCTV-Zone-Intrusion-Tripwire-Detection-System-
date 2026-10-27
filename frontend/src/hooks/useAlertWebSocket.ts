import { useEffect, useRef, useState, useCallback } from 'react';
import type { WSAlertMessage } from '../types';

export function useAlertWebSocket(onAlert?: (msg: WSAlertMessage) => void) {
  const [isConnected, setIsConnected] = useState(false);
  const [latestAlert, setLatestAlert] = useState<WSAlertMessage | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeout = useRef<number | null>(null);

  const connect = useCallback(() => {
    try {
      const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
      const wsUrl = `${protocol}//${window.location.host}/ws/alerts`;
      const ws = new WebSocket(wsUrl);

      ws.onopen = () => {
        setIsConnected(true);
        console.log('[Tripwire WS] Connected to live alarms stream');
      };

      ws.onmessage = (event) => {
        try {
          const data: WSAlertMessage = JSON.parse(event.data);
          setLatestAlert(data);
          if (onAlert) {
            onAlert(data);
          }
        } catch (e) {
          console.error('[Tripwire WS] Parse error:', e);
        }
      };

      ws.onclose = () => {
        setIsConnected(false);
        // Auto-reconnect with 3s backoff
        reconnectTimeout.current = window.setTimeout(() => {
          connect();
        }, 3000);
      };

      ws.onerror = (err) => {
        console.warn('[Tripwire WS] Error:', err);
        ws.close();
      };

      wsRef.current = ws;
    } catch (err) {
      console.error('[Tripwire WS] Connection initiation error:', err);
    }
  }, [onAlert]);

  useEffect(() => {
    connect();
    return () => {
      if (reconnectTimeout.current) clearTimeout(reconnectTimeout.current);
      if (wsRef.current) wsRef.current.close();
    };
  }, [connect]);

  return { isConnected, latestAlert };
}
