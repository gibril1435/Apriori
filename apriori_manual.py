import pandas as pd
import itertools

# --- FUNGSI BANTUAN (SESUAI GAMBAR) ---

# Langkah 1 & 3: Menghitung Nilai Support
# Rumus Gambar: Support(A) = Count(A) / N
def calculate_support(itemset, transactions):
    count = 0
    for transaction in transactions:
        if itemset.issubset(transaction):
            count += 1
    N = len(transactions)
    return count / N

# Langkah 2: Membentuk Kandidat Itemset (Ck)
# Rumus Gambar: C_k = F_k-1 JOIN F_k-1
def generate_candidates(prev_frequent_itemsets, k):
    candidates = set()
    # Menggabungkan itemset (Self-Join)
    for itemset1 in prev_frequent_itemsets:
        for itemset2 in prev_frequent_itemsets:
            # Gabungkan dua itemset
            union_set = itemset1.union(itemset2)
            # Kita hanya ingin kandidat dengan panjang k
            if len(union_set) == k:
                candidates.add(union_set)
    return list(candidates)

# Langkah 4: Melakukan Pruning (Apriori Property)
# Gambar: "Jika A tidak frequent, maka superset A dieliminasi"
def pruning(candidates, prev_frequent_itemsets, k):
    pruned_candidates = []
    for candidate in candidates:
        is_valid = True
        # Cek semua subset dari kandidat
        subsets = itertools.combinations(candidate, k-1)
        for subset in subsets:
            # Jika ada subset yang TIDAK ditemukan di itemset frequent sebelumnya
            if frozenset(subset) not in prev_frequent_itemsets:
                is_valid = False # Prune (pangkas) kandidat ini
                break
        if is_valid:
            pruned_candidates.append(candidate)
    return pruned_candidates

# --- ALGORITMA UTAMA ---

def apriori_manual(transactions, min_support):
    # Data harus dalam bentuk list of sets untuk operasi matematika himpunan
    trans_sets = [set(t) for t in transactions]
    
    # 1. Mencari Frequent Itemset level-1 (C1 -> L1)
    items = set()
    for t in trans_sets:
        items.update(t)
    
    frequent_itemsets = {} # Menyimpan semua itemset yang lolos
    level_frequent = [] # L1, L2, dst
    
    # Hitung support untuk 1 item
    current_l = []
    for item in items:
        itemset = frozenset([item])
        supp = calculate_support(itemset, trans_sets)
        if supp >= min_support:
            current_l.append(itemset)
            frequent_itemsets[itemset] = supp
            
    level_frequent.append(set(current_l))
    
    k = 2
    while len(current_l) > 0:
        # Langkah 2: Generate Candidate Ck
        candidates = generate_candidates(current_l, k)
        
        # Langkah 4: Pruning
        candidates_pruned = pruning(candidates, level_frequent[-1], k)
        
        # Langkah 3: Hitung Support Kandidat yang tersisa
        next_l = []
        for candidate in candidates_pruned:
            supp = calculate_support(candidate, trans_sets)
            # Filter Support >= MinSupport
            if supp >= min_support:
                next_l.append(candidate)
                frequent_itemsets[candidate] = supp
        
        if not next_l:
            break
            
        level_frequent.append(set(next_l))
        current_l = next_l
        k += 1
        
    return frequent_itemsets

# Langkah 5: Menghasilkan Aturan Asosiasi (Confidence & Lift)
def generate_rules(frequent_itemsets, min_confidence, transactions_len):
    rules = []
    for itemset, support_ab in frequent_itemsets.items():
        if len(itemset) > 1:
            # Buat semua kemungkinan kombinasi (A -> B)
            for i in range(1, len(itemset)):
                for antecedent in itertools.combinations(itemset, i):
                    antecedent = frozenset(antecedent)
                    consequent = itemset.difference(antecedent)
                    
                    support_a = frequent_itemsets[antecedent]
                    support_b = frequent_itemsets[consequent]
                    
                    # Rumus Confidence (Sesuai Gambar)
                    # Conf(A->B) = Support(A,B) / Support(A)
                    confidence = support_ab / support_a
                    
                    # Rumus Lift (Sesuai Gambar)
                    # Lift(A->B) = Support(A,B) / (Support(A) * Support(B))
                    lift = support_ab / (support_a * support_b)
                    
                    if confidence >= min_confidence:
                        rules.append({
                            'Antecedent': list(antecedent),
                            'Consequent': list(consequent),
                            'Support': round(support_ab, 4),
                            'Confidence': round(confidence, 4),
                            'Lift': round(lift, 4)
                        })
    return pd.DataFrame(rules)

# --- EKSEKUSI PROGRAM ---

# 1. Load Data (Contoh memotong 100 data agar cepat dipahami logikanya)
# Ganti 100 dengan len(full_dataset) jika ingin semua data
full_dataset = pd.read_csv('Market_Basket_Optimisation.csv', header=None)
subset_data = full_dataset.head(100) 

print("Memproses Data...")
transactions = []
for i in range(0, len(subset_data)):
    # Bersihkan 'nan'
    transactions.append([str(subset_data.values[i,j]) for j in range(0, 20) if str(subset_data.values[i,j]) != 'nan'])

# 2. Jalankan Apriori
# Min Support 5% (0.05), Min Confidence 20% (0.2)
freq_items = apriori_manual(transactions, min_support=0.05)
rules_df = generate_rules(freq_items, min_confidence=0.2, transactions_len=len(transactions))

# 3. Tampilkan Hasil
print("\n=== Frequent Itemsets (Contoh) ===")
for k, v in list(freq_items.items())[:5]:
    print(f"Item: {list(k)}, Support: {v:.4f}")

print("\n=== Association Rules (Sesuai Langkah 5) ===")
# Urutkan berdasarkan Lift tertinggi
print(rules_df.sort_values(by='Lift', ascending=False).head(10).to_string(index=False))