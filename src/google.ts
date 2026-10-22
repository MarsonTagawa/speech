// Picks the file out of a Google Slides / Docs / Sheets / Drive share link.
// Rust (drive.rs) builds the download from it: Docs-type files export to PDF;
// a Drive file downloads as-is, so it has to be a PDF already.
export type GoogleKind = "presentation" | "document" | "spreadsheets" | "file";

export function googleFile(link: string): { kind: GoogleKind; id: string } | null {
  const doc = link.match(/docs\.google\.com\/(presentation|document|spreadsheets)\/d\/([\w-]+)/);
  if (doc) return { kind: doc[1] as GoogleKind, id: doc[2] };
  const file = link.match(/drive\.google\.com\/(?:file\/d\/|open\?id=|uc\?(?:.*&)?id=)([\w-]+)/);
  return file ? { kind: "file", id: file[1] } : null;
}
