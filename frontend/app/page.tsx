"use client";

import { useState, useEffect, useRef } from "react";
import { supabase } from "@/lib/supabase";

export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [category, setCategory] = useState("shirt");
  const [items, setItems] = useState<any[]>([]);
  const [loading, setLoading] = useState(false);
  const [outfits, setOutfits] = useState<any[]>([]);
  const [uploadError, setUploadError] = useState<string | null>(null);

  const fileInputRef = useRef<HTMLInputElement>(null);

  // ---------------- FETCH ----------------
  const fetchItems = async () => {
    const { data } = await supabase
      .from("wardrobe_items")
      .select("*")
      .order("created_at", { ascending: false });
    setItems(data || []);
  };

  useEffect(() => {
    fetchItems();
  }, []);

  // ---------------- UPLOAD ----------------
  const handleUpload = async () => {
    if (!file) return;

    setLoading(true);
    setUploadError(null);

    try {
      const formData = new FormData();
      formData.append("image", file);

      // ✅ Check if backend is reachable + handle errors properly
      const res = await fetch("http://127.0.0.1:5000/analyze", {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.error || `Server error: ${res.status}`);
      }

      const { color, category: detectedCategory, confidence } = await res.json();

      console.log(`Detected: ${detectedCategory} (${(confidence * 100).toFixed(1)}%) — Color: ${color}`);

      const finalCategory = detectedCategory || category;

      // Upload image to Supabase storage
      const fileName = `${Date.now()}-${file.name}`;
      const { error: uploadErr } = await supabase.storage
        .from("wardrobe-images")
        .upload(fileName, file);

      if (uploadErr) throw uploadErr;

      const { data } = supabase.storage
        .from("wardrobe-images")
        .getPublicUrl(fileName);

      // Save to DB
      await supabase.from("wardrobe_items").insert([
        {
          image_url: data.publicUrl,
          category: finalCategory,
          color: color,
        },
      ]);

      await fetchItems();

      setFile(null);
      setCategory("shirt");
      if (fileInputRef.current) fileInputRef.current.value = "";

    } catch (err: any) {
      console.error(err);
      setUploadError(err.message || "Upload failed. Is the Flask server running?");
    }

    setLoading(false);
  };

  // ---------------- CATEGORY GROUPING ----------------
  const getGroup = (cat: string) => {
    if (cat === "shirt") return "top";
    if (cat === "jeans") return "bottom";
    if (cat === "dress") return "full";
    if (cat === "shoes") return "footwear";
    return "unknown";
  };

  // ---------------- COLOR SCORE ----------------
  const getColorScore = (c1: string, c2: string) => {
    const neutral = ["black", "white", "gray"];
    if (c1 === c2) return 3;
    if (neutral.includes(c1) || neutral.includes(c2)) return 2;
    const goodPairs: Record<string, string[]> = {
      blue: ["white", "black", "gray"],
      red: ["black", "white", "gray"],
      green: ["white", "black"],
      yellow: ["black", "blue"],
    };
    if (goodPairs[c1]?.includes(c2)) return 2;
    return 1; // ✅ was 0 — now gives baseline score to all combos
  };

  // ---------------- OUTFIT GENERATION ----------------
  const generateOutfits = () => {
    const tops     = items.filter(i => getGroup(i.category) === "top");
    const bottoms  = items.filter(i => getGroup(i.category) === "bottom");
    const dresses  = items.filter(i => getGroup(i.category) === "full");
    const shoes    = items.filter(i => getGroup(i.category) === "footwear");

    const results: any[] = [];

    // 👕 Top + Bottom + Shoes
    tops.forEach(top => {
      bottoms.forEach(bottom => {
        shoes.forEach(shoe => {
          const score =
            getColorScore(top.color, bottom.color) +
            getColorScore(top.color, shoe.color) +
            getColorScore(bottom.color, shoe.color);

          results.push({ top, bottom, shoe, score, reason: "Top + Bottom + Shoes" });
        });
      });
    });

    // 👗 Dress + Shoes
    dresses.forEach(dress => {
      shoes.forEach(shoe => {
        const score = getColorScore(dress.color, shoe.color);
        results.push({ dress, shoe, score, reason: "Dress + Shoes" });
      });
    });

    // 🔄 Fallback — if no structured combos possible
    if (results.length === 0) {
      items.forEach(a => {
        items.forEach(b => {
          if (a.id !== b.id) {
            results.push({ top: a, bottom: b, score: 1, reason: "Fallback combo" });
          }
        });
      });
    }

    results.sort((a, b) => b.score - a.score);
    setOutfits(results.slice(0, 6));
  };

  const handleDelete = async (item: any) => {
    // Extract filename from URL
    const fileName = item.image_url.split("/").pop();

    // Delete from storage
    await supabase.storage.from("wardrobe-images").remove([fileName]);

    // Delete from DB
    await supabase.from("wardrobe_items").delete().eq("id", item.id);

    await fetchItems();
  };

  return (
    <div className="p-10 space-y-8">
      <h1 className="text-3xl font-bold">👗 My Wardrobe</h1>

      {/* Upload */}
      <div className="flex flex-col gap-3 max-w-sm">
        <input
          ref={fileInputRef}
          type="file"
          accept="image/*"
          onChange={(e) => setFile(e.target.files?.[0] || null)}
          className="border p-2 rounded"
        />

        <select
          value={category}
          onChange={(e) => setCategory(e.target.value)}
          className="border p-2 rounded"
        >
          <option value="shirt">Shirt</option>
          <option value="jeans">Jeans</option>
          <option value="dress">Dress</option>
          <option value="shoes">Shoes</option>
        </select>

        <button
          onClick={handleUpload}
          disabled={loading || !file}
          className="bg-black text-white px-4 py-2 rounded disabled:opacity-50"
        >
          {loading ? "Uploading..." : "Upload"}
        </button>

        {/* ✅ Show error if upload fails */}
        {uploadError && (
          <p className="text-red-500 text-sm">{uploadError}</p>
        )}
      </div>

      {/* Generate Outfits */}
      <button
        onClick={generateOutfits}
        disabled={items.length < 2}
        className="bg-gray-800 text-white px-4 py-2 rounded disabled:opacity-50"
      >
        Generate Outfits
      </button>

      {/* Wardrobe Grid */}
      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-6">
        {items.map((item) => (
          <div key={item.id} className="border rounded-xl p-2 relative group">  
            {/* ✅ This will now show on hover */}
            <button
              onClick={() => handleDelete(item)}
              className="absolute top-2 right-2 bg-gray-600 text-white rounded-full w-6 h-6 
                        items-center justify-center text-xs font-bold z-10"
            >
              ✕
            </button>

            <img
              src={item.image_url}
              className="w-full h-40 object-contain bg-gray-50"
              alt={item.category}
            />
            <p className="text-center mt-2 capitalize font-medium">{item.category}</p>
            <p className="text-center text-sm text-gray-500 capitalize">{item.color}</p>
          </div>
        ))}
      </div>

      {/* Outfits */}
      {outfits.length > 0 && (
        <div className="mt-10">
          <h2 className="text-2xl font-bold mb-4">✨ Generated Outfits</h2>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {outfits.map((outfit, index) => (
              <div key={index} className="border p-4 rounded-xl space-y-2">

                {/* ✅ Each piece checked individually before rendering */}
                {outfit.top && (
                  <img src={outfit.top.image_url} className="h-32 object-contain w-full" alt="top" />
                )}
                {outfit.bottom && (
                  <img src={outfit.bottom.image_url} className="h-32 object-contain w-full" alt="bottom" />
                )}
                {outfit.dress && (
                  <img src={outfit.dress.image_url} className="h-32 object-contain w-full" alt="dress" />
                )}
                {outfit.shoe && (
                  <img src={outfit.shoe.image_url} className="h-32 object-contain w-full" alt="shoes" />
                )}

                <div className="flex justify-between items-center">
                  <p className="text-sm text-gray-600">{outfit.reason}</p>
                  <p className="text-sm font-semibold">Score: {outfit.score}</p>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}