/*
World Fire Propagation Map - Service Worker
Provides offline caching and background sync for critical alerts.
*/

const CACHE_NAME = 'firemap-v3-cache';
const STATIC_CACHE = 'firemap-static-v3';
const API_CACHE = 'firemap-api-v3';

// Critical resources to cache
const STATIC_ASSETS = [
    '/',
    '/manifest.json'
];

// API endpoints to cache
const API_ROUTES = [
    '/api/v1/fires',
    '/api/v1/weather',
    '/api/v1/risk'
];

// Install event - cache static assets
self.addEventListener('install', (event) => {
    console.log('[SW] Installing service worker...');
    
    event.waitUntil(
        caches.open(STATIC_CACHE)
            .then((cache) => {
                console.log('[SW] Caching static assets');
                return cache.addAll(STATIC_ASSETS);
            })
            .then(() => self.skipWaiting())
    );
});

// Activate event - clean old caches
self.addEventListener('activate', (event) => {
    console.log('[SW] Activating service worker...');
    
    event.waitUntil(
        caches.keys()
            .then((cacheNames) => {
                return Promise.all(
                    cacheNames
                        .filter((name) => name !== STATIC_CACHE && name !== API_CACHE)
                        .map((name) => {
                            console.log('[SW] Deleting old cache:', name);
                            return caches.delete(name);
                        })
                );
            })
            .then(() => self.clients.claim())
    );
});

// Fetch event - serve from cache, fallback to network
self.addEventListener('fetch', (event) => {
    const { request } = event;
    const url = new URL(request.url);
    
    // Skip non-GET requests
    if (request.method !== 'GET') {
        return;
    }
    
    // API requests - network first, cache fallback
    if (url.pathname.startsWith('/api/')) {
        event.respondWith(networkFirst(request, API_CACHE));
        return;
    }
    
    // Static assets - cache first
    event.respondWith(cacheFirst(request, STATIC_CACHE));
});

// Network first strategy
async function networkFirst(request, cacheName) {
    try {
        const networkResponse = await fetch(request);
        
        // Cache successful responses
        if (networkResponse.ok) {
            const cache = await caches.open(cacheName);
            cache.put(request, networkResponse.clone());
        }
        
        return networkResponse;
    } catch (error) {
        // Fallback to cache
        const cachedResponse = await caches.match(request);
        if (cachedResponse) {
            return cachedResponse;
        }
        
        // Return offline response for API
        return new Response(
            JSON.stringify({
                error: "offline",
                message: "You are offline. Data may be stale."
            }),
            { status: 503, headers: { "Content-Type": "application/json" } }
        );
    }
}

// Cache first strategy
async function cacheFirst(request, cacheName) {
    const cachedResponse = await caches.match(request);
    
    if (cachedResponse) {
        return cachedResponse;
    }
    
    try {
        const networkResponse = await fetch(request);
        
        if (networkResponse.ok) {
            const cache = await caches.open(cacheName);
            cache.put(request, networkResponse.clone());
        }
        
        return networkResponse;
    } catch (error) {
        return new Response('Offline', { status: 503 });
    }
}

// Background sync for critical alerts
self.addEventListener('sync', (event) => {
    console.log('[SW] Background sync:', event.tag);
    
    if (event.tag === 'sync-alerts') {
        event.waitUntil(syncAlerts());
    }
});

async function syncAlerts() {
    // Get pending alerts from IndexedDB and sync
    console.log('[SW] Syncing critical alerts...');
}

// Push notification handling
self.addEventListener('push', (event) => {
    console.log('[SW] Push received:', event);
    
    const data = event.data?.json() || {
        title: 'Fire Alert',
        body: 'Check the app for details',
        icon: '/manifest.json'
    };
    
    event.waitUntil(
        self.registration.showNotification(data.title, {
            body: data.body,
            icon: data.icon,
            badge: '/manifest.json',
            tag: 'fire-alert',
            requireInteraction: data.severity === 'critical'
        })
    );
});

// Notification click handling
self.addEventListener('notificationclick', (event) => {
    console.log('[SW] Notification clicked:', event);
    
    event.notification.close();
    
    event.waitUntil(
        clients.matchAll({ type: 'window', includeUncontrolled: true })
            .then((clientList) => {
                // Focus existing window or open new
                for (const client of clientList) {
                    if (client.url.includes('/') && 'focus' in client) {
                        return client.focus();
                    }
                }
                
                if (clients.openWindow) {
                    return clients.openWindow('/');
                }
            })
    );
});

// Message handling from main app
self.addEventListener('message', (event) => {
    console.log('[SW] Message received:', event.data);
    
    if (event.data.type === 'SKIP_WAITING') {
        self.skipWaiting();
    }
    
    if (event.data.type === 'CACHE_API') {
        event.waitUntil(cacheApiResponse(event.data.url));
    }
});

async function cacheApiResponse(url) {
    try {
        const response = await fetch(url);
        if (response.ok) {
            const cache = await caches.open(API_CACHE);
            await cache.put(url, response);
            console.log('[SW] Cached API response:', url);
        }
    } catch (error) {
        console.log('[SW] Failed to cache API response:', error);
    }
}
