"use client";
 
import { useState, useEffect, useRef } from "react";
import { supabase } from "@/lib/supabase";
 
// ── CONSTANTS ──────────────────────────────────────────────────────────────
 
const CATEGORY_LABEL: Record<string, string> = {
  shirt: "Top",
  jeans: "Bottom",
  dress: "Dress",
  shoes: "Shoes",
};
 
const CATEGORY_BG: Record<string, string> = {
  shirt: "bg-purple-50",
  jeans: "bg-blue-50",
  dress: "bg-pink-50",
  shoes: "bg-green-50",
};
 
const COLOR_DOT: Record<string, string> = {
  white: "bg-white border border-gray-300",
  black: "bg-gray-900",
  gray: "bg-gray-400",
  red: "bg-red-500",
  green: "bg-green-500",
  blue: "bg-blue-500",
  yellow: "bg-yellow-400",
  purple: "bg-purple-500",
  mixed: "bg-gradient-to-br from-pink-400 via-yellow-300 to-blue-400",
};
 
// ── [NEW] COLOR MATCHING ───────────────────────────────────────────────────
// Maps each color to colors it pairs well with
const COLOR_MATCH: Record<string, string[]> = {
  black:  ["white", "gray", "blue", "red", "yellow", "purple", "green", "mixed"],
  white:  ["black", "gray", "blue", "red", "green", "yellow", "purple", "mixed"],
  gray:   ["black", "white", "blue", "red", "purple"],
  blue:   ["white", "gray", "black", "yellow"],
  red:    ["black", "white", "gray"],
  green:  ["white", "black", "yellow"],
  yellow: ["black", "blue", "green"],
  purple: ["white", "black", "gray"],
  mixed:  ["black", "white", "gray"],
};
 
// Returns true if two colors are compatible
const isColorMatch = (c1: string, c2: string): boolean => {
  if (!c1 || !c2) return true; // if color unknown, allow it
  if (c1 === c2) return true;  // same color always matches
  return COLOR_MATCH[c1]?.includes(c2) || COLOR_MATCH[c2]?.includes(c1) || false;
};
 
// ── [NEW] OCCASION / STYLE SYSTEM ─────────────────────────────────────────
type Occasion = "casual" | "party" | "office";
 
// Which item styles are appropriate for each occasion
const OCCASION_STYLES: Record<Occasion, string[]> = {
  casual: ["casual", "everyday", "relaxed", "streetwear"],
  party:  ["party", "evening", "formal", "chic", "glamour"],
  office: ["office", "formal", "business", "smart", "professional"],
};
 
// Returns true if an item's style fits the selected occasion
// Falls back to "casual" if item has no style field
const isStyleMatch = (itemStyle: string | undefined, occasion: Occasion): boolean => {
  const style = (itemStyle || "casual").toLowerCase();
  return OCCASION_STYLES[occasion].includes(style);
};
 
// ── [NEW] OUTFIT REASON BUILDER ────────────────────────────────────────────
const buildReason = (occasion: Occasion, topColor: string, bottomColor: string): string => {
  const colorNote = topColor === bottomColor
    ? "tonal color pairing"
    : "complementary colors";
 
  const occasionLabel = {
    casual: "casual",
    party:  "evening",
    office: "office",
  }[occasion];
 
  return `${occasionLabel} look · ${colorNote}`;
};
 
// ── TYPES ──────────────────────────────────────────────────────────────────
interface WardrobeItem {
  id: string;
  image_url: string;
  category: string;
  color: string;
  style?: string;
  created_at: string;
}
 
interface Outfit {
  top?: WardrobeItem;
  bottom?: WardrobeItem;
  dress?: WardrobeItem;
  shoe?: WardrobeItem;
  reason: string;
  score: number;
}
 
// ── COMPONENT ──────────────────────────────────────────────────────────────
export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [dragOver, setDragOver] = useState(false);
  const [items, setItems] = useState<WardrobeItem[]>([]);
  const [loading, setLoading] = useState(false);
  const [outfits, setOutfits] = useState<Outfit[]>([]);
  const [likedOutfits, setLikedOutfits] = useState<Record<number, boolean>>({});
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<"wardrobe" | "looks">("wardrobe");
 
  // [NEW] Occasion state
  const [occasion, setOccasion] = useState<Occasion>("casual");
 
  const fileInputRef = useRef<HTMLInputElement>(null);
 
  // ── FETCH ────────────────────────────────────────────────────────────────
  const fetchItems = async () => {
    const { data } = await supabase
      .from("wardrobe_items")
      .select("*")
      .order("created_at", { ascending: false });
    setItems((data as WardrobeItem[]) || []);
  };
 
  useEffect(() => { fetchItems(); }, []);
 
  // ── FILE HANDLING ─────────────────────────────────────────────────────────
  const handleFileChange = (f: File | null) => {
    setFile(f);
    setPreview(f ? URL.createObjectURL(f) : null);
  };
 
  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    const f = e.dataTransfer.files?.[0];
    if (f) handleFileChange(f);
  };
 
  // ── UPLOAD ────────────────────────────────────────────────────────────────
  // [NEW] Wake server before uploading (handles Render free tier spin-down)
  const wakeUpServer = async (): Promise<boolean> => {
    const BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:5000";
    for (let i = 0; i < 10; i++) {
      try {
        const res = await fetch(`${BASE_URL}/`, {
          signal: AbortSignal.timeout(5000),
        });
        if (res.ok) return true;
      } catch {}
      await new Promise(r => setTimeout(r, 3000));
    }
    return false;
  };
 
  const handleUpload = async () => {
    if (!file) return;
    setLoading(true);
    setUploadError(null);
 
    try {
      const BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:5000";
 
      // Wake server first
      setUploadError("⏳ Waking up server, please wait...");
      const awake = await wakeUpServer();
      if (!awake) throw new Error("Server took too long to respond. Try again in a moment.");
      setUploadError(null);
 
      const formData = new FormData();
      formData.append("image", file);
 
      const res = await fetch(`${BASE_URL}/analyze`, {
        method: "POST",
        body: formData,
      });
 
      if (!res.ok) throw new Error(`Server error: ${res.status}`);
      const { color, category } = await res.json();
 
      const fileName = `${Date.now()}-${file.name}`;
      await supabase.storage.from("wardrobe-images").upload(fileName, file);
 
      const { data } = supabase.storage
        .from("wardrobe-images")
        .getPublicUrl(fileName);
 
      await supabase.from("wardrobe_items").insert([
        { image_url: data.publicUrl, category, color },
      ]);
 
      await fetchItems();
      setFile(null);
      setPreview(null);
      if (fileInputRef.current) fileInputRef.current.value = "";
    } catch (err: any) {
      setUploadError(err.message || "Upload failed.");
    }
 
    setLoading(false);
  };
 
  // ── CATEGORY GROUPING ─────────────────────────────────────────────────────
  const getGroup = (cat: string) => {
    if (cat === "shirt") return "top";
    if (cat === "jeans") return "bottom";
    if (cat === "dress") return "full";
    if (cat === "shoes") return "footwear";
    return "unknown";
  };
 
  // ── [NEW] SMART OUTFIT GENERATION ─────────────────────────────────────────
  // Replaces the old random pickRandom approach with scored, filtered generation
  const generateOutfits = () => {
    const tops    = items.filter(i => getGroup(i.category) === "top");
    const bottoms = items.filter(i => getGroup(i.category) === "bottom");
    const dresses = items.filter(i => getGroup(i.category) === "full");
    const shoes   = items.filter(i => getGroup(i.category) === "footwear");
 
    const results: Outfit[] = [];
 
    // ── DRESS + SHOES ──
    dresses.forEach(dress => {
      shoes.forEach(shoe => {
        // Style check
        const styleOk = isStyleMatch(dress.style, occasion) &&
                        isStyleMatch(shoe.style, occasion);
        if (!styleOk) return;
 
        // Color check
        const colorOk = isColorMatch(dress.color, shoe.color);
        if (!colorOk) return;
 
        // Score: same color = 1, complementary = 2
        const score = dress.color === shoe.color ? 1 : 2;
 
        results.push({
          dress,
          shoe,
          score,
          reason: `${occasion} look · dress & shoes`,
        });
      });
    });
 
    // ── TOP + BOTTOM + SHOES ──
    tops.forEach(top => {
      bottoms.forEach(bottom => {
        shoes.forEach(shoe => {
          // Style check — all 3 pieces must match occasion
          const styleOk = isStyleMatch(top.style, occasion) &&
                          isStyleMatch(bottom.style, occasion) &&
                          isStyleMatch(shoe.style, occasion);
          if (!styleOk) return;
 
          // Color check — top+bottom and bottom+shoe must be compatible
          const colorOk = isColorMatch(top.color, bottom.color) &&
                          isColorMatch(bottom.color, shoe.color);
          if (!colorOk) return;
 
          // Score based on how many pairs match
          let score = 0;
          if (isColorMatch(top.color, bottom.color)) score += 2;
          if (isColorMatch(bottom.color, shoe.color)) score += 2;
          if (isColorMatch(top.color, shoe.color))    score += 1;
 
          results.push({
            top,
            bottom,
            shoe,
            score,
            reason: buildReason(occasion, top.color, bottom.color),
          });
        });
      });
    });
 
    // ── FALLBACK: if style filter is too strict, relax it ──
    // If nothing was generated, ignore style and just use color
    if (results.length === 0) {
      tops.forEach(top => {
        bottoms.forEach(bottom => {
          shoes.forEach(shoe => {
            if (!isColorMatch(top.color, bottom.color)) return;
            if (!isColorMatch(bottom.color, shoe.color)) return;
            results.push({
              top, bottom, shoe,
              score: 1,
              reason: `${occasion} look · color coordinated`,
            });
          });
        });
      });
 
      dresses.forEach(dress => {
        shoes.forEach(shoe => {
          if (!isColorMatch(dress.color, shoe.color)) return;
          results.push({
            dress, shoe,
            score: 1,
            reason: `${occasion} look · dress & shoes`,
          });
        });
      });
    }
 
    // ── LAST RESORT: no color filter either ──
    if (results.length === 0 && items.length >= 2) {
      tops.forEach(top => {
        bottoms.forEach(bottom => {
          shoes.forEach(shoe => {
            results.push({ top, bottom, shoe, score: 0, reason: "mix & match" });
          });
        });
      });
    }
 
    // Sort by score descending, deduplicate, limit to 6
    results.sort((a, b) => b.score - a.score || Math.random() - 0.5);
 
    const unique: Outfit[] = [];
    const seen = new Set<string>();
    for (const o of results) {
      const key = JSON.stringify([
        o.top?.color,
        o.bottom?.color,
        o.dress?.color,
        o.shoe?.color
      ]);
      if (!seen.has(key)) {
        seen.add(key);
        unique.push(o);
      }
    }
 
    setOutfits(unique.slice(0, 10));
    setLikedOutfits({});
    setActiveTab("looks");
  };
 
  // ── [NEW] SAVE OUTFIT TO SUPABASE ─────────────────────────────────────────
  // Replaces old handleFeedback — now saves full outfit_data to saved_outfits table
  const handleFeedback = async (index: number, liked: boolean) => {
    setLikedOutfits(prev => ({ ...prev, [index]: liked }));
 
    const outfit = outfits[index];
 
    // Build a clean serializable version of the outfit
    const outfitData = {
      top:    outfit.top    ? { id: outfit.top.id,    image_url: outfit.top.image_url,    category: outfit.top.category,    color: outfit.top.color }    : null,
      bottom: outfit.bottom ? { id: outfit.bottom.id, image_url: outfit.bottom.image_url, category: outfit.bottom.category, color: outfit.bottom.color } : null,
      dress:  outfit.dress  ? { id: outfit.dress.id,  image_url: outfit.dress.image_url,  category: outfit.dress.category,  color: outfit.dress.color }  : null,
      shoe:   outfit.shoe   ? { id: outfit.shoe.id,   image_url: outfit.shoe.image_url,   category: outfit.shoe.category,   color: outfit.shoe.color }   : null,
      reason: outfit.reason,
      occasion,
    };
 
    // [NEW] Save to saved_outfits table
    await supabase.from("saved_outfits").insert([
      {
        outfit_data: outfitData,
        liked,
      },
    ]);
  };
 
  // ── DELETE ────────────────────────────────────────────────────────────────
  const handleDelete = async (item: WardrobeItem) => {
    const fileName = item.image_url.split("/").pop();
    await supabase.storage.from("wardrobe-images").remove([fileName!]);
    await supabase.from("wardrobe_items").delete().eq("id", item.id);
    await fetchItems();
  };
 
  // ── RENDER ────────────────────────────────────────────────────────────────
  return (
    <div className="min-h-screen bg-stone-50 text-stone-900">
 
      {/* ── HEADER ── */}
      <header className="sticky top-0 z-50 bg-stone-50/90 backdrop-blur-sm border-b border-stone-200 px-6 md:px-12 py-4 flex items-center justify-between gap-4">
 
        <h1 className="text-xl font-semibold tracking-widest uppercase text-stone-800">
          Style<span className="text-amber-900">.</span>Sense
        </h1>
 
        {/* Tab switcher */}
        <div className="flex items-center bg-stone-200 rounded-full p-1 gap-1">
          <button
            onClick={() => setActiveTab("wardrobe")}
            className={`px-5 py-1.5 rounded-full text-xs font-medium tracking-wide transition-all duration-200 ${
              activeTab === "wardrobe"
                ? "bg-stone-900 text-white shadow-sm"
                : "text-stone-500 hover:text-stone-800"
            }`}
          >
            Wardrobe {items.length > 0 && `(${items.length})`}
          </button>
          <button
            onClick={() => setActiveTab("looks")}
            className={`px-5 py-1.5 rounded-full text-xs font-medium tracking-wide transition-all duration-200 ${
              activeTab === "looks"
                ? "bg-stone-900 text-white shadow-sm"
                : "text-stone-500 hover:text-stone-800"
            }`}
          >
            Looks {outfits.length > 0 && `(${outfits.length})`}
          </button>
        </div>
 
        {/* [NEW] Occasion selector + Generate button grouped together */}
        <div className="flex items-center gap-3">
          <select
            value={occasion}
            onChange={(e) => setOccasion(e.target.value as Occasion)}
            className="text-xs font-medium text-stone-600 bg-white border border-stone-200 rounded-full px-4 py-2 outline-none cursor-pointer hover:border-stone-400 transition-all"
          >
            <option value="casual">👟 Casual</option>
            <option value="party">🎉 Party</option>
            <option value="office">💼 Office</option>
          </select>
 
          <button
            onClick={generateOutfits}
            disabled={items.length < 2}
            className="flex items-center gap-2 px-5 py-2 bg-amber-800 hover:bg-amber-600 disabled:opacity-40 disabled:cursor-not-allowed text-white text-xs font-semibold tracking-wider uppercase rounded-full transition-all duration-200 hover:-translate-y-0.5 hover:shadow-lg hover:shadow-amber-200"
          >
            <span>✦</span> Generate Looks
          </button>
        </div>
      </header>
 
      <main className="max-w-7xl mx-auto px-6 md:px-12 py-10 space-y-14">
 
        {/* ════════════════════════════════
            TAB: WARDROBE
        ════════════════════════════════ */}
        {activeTab === "wardrobe" && (
          <>
            {/* Upload section */}
            <div className="grid md:grid-cols-2 gap-8 items-center bg-white rounded-2xl border border-stone-200 p-8">
 
              {/* Drop zone */}
              <div
                onClick={() => fileInputRef.current?.click()}
                onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
                onDragLeave={() => setDragOver(false)}
                onDrop={handleDrop}
                className={`cursor-pointer rounded-xl border-2 border-dashed min-h-50 flex flex-col items-center justify-center gap-3 transition-all duration-200
                  ${dragOver
                    ? "border-amber-400 bg-amber-50"
                    : preview
                    ? "border-amber-300 bg-stone-50"
                    : "border-stone-300 bg-stone-50 hover:border-stone-400 hover:bg-stone-100"
                  }`}
              >
                <input
                  ref={fileInputRef}
                  type="file"
                  accept="image/*"
                  className="hidden"
                  onChange={(e) => handleFileChange(e.target.files?.[0] || null)}
                />
                {preview ? (
                  <img src={preview} alt="preview" className="max-h-44 max-w-full object-contain rounded-lg px-4" />
                ) : (
                  <>
                    <div className="w-12 h-12 rounded-full bg-stone-200 flex items-center justify-center text-xl text-stone-400">+</div>
                    <p className="text-sm text-stone-500 font-medium">Drop your item here</p>
                    <p className="text-xs text-stone-400">or click to browse · JPG, PNG, WEBP</p>
                  </>
                )}
              </div>
 
              {/* Upload CTA */}
              <div className="flex flex-col gap-5">
                <div>
                  <h2 className="text-3xl font-light text-stone-800 leading-snug mb-2">
                    Add to your <span className="italic text-amber-900">wardrobe</span>
                  </h2>
                  <p className="text-sm text-stone-500 leading-relaxed">
                    Our AI detects the category — shirt, jeans, dress, or shoes — and color automatically. Just drop a photo and we handle the rest.
                  </p>
                </div>
 
                <button
                  onClick={handleUpload}
                  disabled={!file || loading}
                  className="self-start flex items-center gap-2 px-7 py-3 bg-stone-900 hover:bg-stone-700 disabled:opacity-40 disabled:cursor-not-allowed text-white text-sm font-medium rounded-full transition-all duration-200 hover:-translate-y-0.5"
                >
                  {loading ? (
                    <>
                      <span className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin inline-block" />
                      Analysing...
                    </>
                  ) : "Add to Wardrobe"}
                </button>
 
                {uploadError && (
                  <div className="flex items-start gap-2 bg-red-50 border border-red-200 text-red-600 text-xs px-4 py-3 rounded-xl">
                    <span>⚠</span>
                    <span>{uploadError}</span>
                  </div>
                )}
              </div>
            </div>
 
            {/* Wardrobe grid */}
            <div>
              <div className="flex items-baseline gap-3 mb-6">
                <h2 className="text-2xl font-light text-stone-800">My Wardrobe</h2>
                {items.length > 0 && (
                  <span className="text-xs text-stone-400 bg-stone-200 px-3 py-1 rounded-full">
                    {items.length} items
                  </span>
                )}
              </div>
 
              {items.length === 0 ? (
                <div className="flex flex-col items-center justify-center py-24 text-center">
                  <span className="text-5xl mb-4 opacity-30">👗</span>
                  <h3 className="text-xl font-light text-stone-600 mb-1">Your wardrobe is empty</h3>
                  <p className="text-sm text-stone-400">Upload your first piece to get started</p>
                </div>
              ) : (
                <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-4">
                  {items.map((item) => (
                    <div
                      key={item.id}
                      className="group relative bg-white rounded-2xl border border-stone-200 overflow-hidden transition-all duration-200 hover:-translate-y-1 hover:shadow-lg hover:shadow-stone-200"
                    >
                      {/* Delete button — visible on hover */}
                      <button
                        onClick={() => handleDelete(item)}
                        className="absolute top-2.5 right-2.5 z-10 w-7 h-7 rounded-full bg-stone-900/70 backdrop-blur-sm text-white text-[10px] flex items-center justify-center opacity-0 group-hover:opacity-100 hover:bg-red-500 transition-all duration-200"
                        title="Remove"
                      >
                        ✕
                      </button>
 
                      <div className={`h-48 flex items-center justify-center p-3 ${CATEGORY_BG[item.category] || "bg-stone-50"}`}>
                        <img src={item.image_url} alt={item.category} className="max-h-full max-w-full object-contain" />
                      </div>
 
                      <div className="px-3 py-2.5 border-t border-stone-100 flex items-center justify-between">
                        <span className="text-[11px] font-semibold uppercase tracking-wider text-stone-700">
                          {CATEGORY_LABEL[item.category] || item.category}
                        </span>
                        <span className="flex items-center gap-1.5 text-[11px] text-stone-400 capitalize">
                          <span className={`w-2.5 h-2.5 rounded-full inline-block shrink-0 ${COLOR_DOT[item.color] || "bg-stone-300"}`} />
                          {item.color}
                        </span>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </>
        )}
 
        {/* ════════════════════════════════
            TAB: LOOKS
        ════════════════════════════════ */}
        {activeTab === "looks" && (
          <div>
            <div className="flex items-baseline gap-3 mb-8">
              <h2 className="text-2xl font-light text-stone-800">Generated Looks</h2>
              {outfits.length > 0 && (
                <span className="text-xs text-stone-400 bg-stone-200 px-3 py-1 rounded-full">
                  {outfits.length} outfits · {occasion}
                </span>
              )}
            </div>
 
            {outfits.length === 0 ? (
              <div className="flex flex-col items-center justify-center py-24 text-center">
                <span className="text-5xl mb-4 opacity-30">✦</span>
                <h3 className="text-xl font-light text-stone-600 mb-1">No looks yet</h3>
                <p className="text-sm text-stone-400 mb-6">Pick an occasion and hit "Generate Looks"</p>
                <button
                  onClick={() => setActiveTab("wardrobe")}
                  className="px-6 py-2.5 bg-stone-900 text-white text-sm rounded-full hover:bg-stone-700 transition"
                >
                  Go to Wardrobe
                </button>
              </div>
            ) : (
              <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-6">
                {outfits.map((outfit, index) => (
                  <div
                    key={index}
                    className="bg-white rounded-2xl border border-stone-200 overflow-hidden transition-all duration-200 hover:-translate-y-1 hover:shadow-lg hover:shadow-stone-100"
                  >
                    {/* Outfit images */}
                    <div className="bg-stone-50 p-4 flex items-center justify-center gap-3 min-h-45 flex-wrap border-b border-stone-100">
                      {outfit.top    && <img src={outfit.top.image_url}    className="h-32 object-contain flex-1 min-w-17.5" alt="top" />}
                      {outfit.bottom && <img src={outfit.bottom.image_url} className="h-32 object-contain flex-1 min-w-17.5" alt="bottom" />}
                      {outfit.dress  && <img src={outfit.dress.image_url}  className="h-32 object-contain flex-1 min-w-17.5" alt="dress" />}
                      {outfit.shoe   && <img src={outfit.shoe.image_url}   className="h-32 object-contain flex-1 min-w-17.5" alt="shoes" />}
                    </div>
 
                    {/* Footer */}
                    <div className="px-4 py-3 flex items-center justify-between gap-2">
                      <span className="text-sm italic text-stone-400 font-light truncate">
                        {outfit.reason}
                      </span>
 
                      {/* [NEW] Like / dislike — saves to saved_outfits */}
                      <div className="flex gap-2 shrink-0">
                        <button
                          onClick={() => handleFeedback(index, true)}
                          title="Love it"
                          className={`w-8 h-8 rounded-full border text-sm flex items-center justify-center transition-all duration-200
                            ${likedOutfits[index] === true
                              ? "bg-stone-900 border-stone-900 text-white"
                              : "border-stone-200 text-stone-400 hover:border-amber-400 hover:text-amber-900"
                            }`}
                        >
                          {likedOutfits[index] === true ? "♥" : "♡"}
                        </button>
                        <button
                          onClick={() => handleFeedback(index, false)}
                          title="Not for me"
                          className={`w-8 h-8 rounded-full border text-xs flex items-center justify-center transition-all duration-200
                            ${likedOutfits[index] === false
                              ? "bg-stone-900 border-stone-900 text-white"
                              : "border-stone-200 text-stone-400 hover:border-stone-400"
                            }`}
                        >
                          ✕
                        </button>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
 
      </main>
    </div>
  );
}