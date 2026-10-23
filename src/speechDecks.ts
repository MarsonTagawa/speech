// Slide files attached to saved speeches, by speech id. Speeches themselves
// live in localStorage, but a deck's PDFs/images can run to megabytes, past its
// quota, so the files go in IndexedDB (which stores File objects as-is).
// Google links don't come through here: the speech keeps the link and the deck
// is downloaded again on open, so edits to the Slides/Doc show up.

const DB = "speech.decks";
const STORE = "files";

let db: Promise<IDBDatabase> | undefined;
function open(): Promise<IDBDatabase> {
  return (db ??= new Promise((ok, fail) => {
    const req = indexedDB.open(DB, 1);
    req.onupgradeneeded = () => req.result.createObjectStore(STORE);
    req.onsuccess = () => ok(req.result);
    req.onerror = () => {
      db = undefined; // let a later call try again
      fail(req.error);
    };
  }));
}

async function run<T>(mode: IDBTransactionMode, op: (s: IDBObjectStore) => IDBRequest<T>): Promise<T> {
  const tx = (await open()).transaction(STORE, mode);
  const req = op(tx.objectStore(STORE));
  return new Promise((ok, fail) => {
    tx.oncomplete = () => ok(req.result);
    tx.onerror = tx.onabort = () => fail(tx.error ?? req.error);
  });
}

export const putDeckFiles = (speech: string, files: File[]) => run("readwrite", (s) => s.put(files, speech)).then(() => {});
export const getDeckFiles = (speech: string) => run<File[] | undefined>("readonly", (s) => s.get(speech));
export const deleteDeckFiles = (speech: string) => run("readwrite", (s) => s.delete(speech)).then(() => {});
