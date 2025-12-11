import Image from "next/image";
import { SimpleThemeToggle } from "@/components/simple-theme-toggle";

export function Header() {
  return (
    <header className="py-16 lg:py-24">
      <div className="flex justify-end gap-4 mb-8">
        <SimpleThemeToggle />
      </div>

      <div className="flex flex-col lg:flex-row items-center lg:items-start gap-8">
        <div className="relative">
          <div className="w-32 h-32 lg:w-40 lg:h-40 rounded-full overflow-hidden border-4 border-primary/20 shadow-xl">
            <Image
              src="/profilepic.jpg"
              alt="Profile"
              width={160}
              height={160}
              className="w-full h-full object-cover"
            />
          </div>
        </div>
        <div className="flex-1 text-center lg:text-left">
          <h1 className="text-4xl lg:text-6xl font-bold bg-gradient-to-r from-primary to-primary/60 bg-clip-text text-transparent mb-6">
            Karthik Reddy Mandhala
          </h1>
          <div className="space-y-2 text-lg lg:text-xl max-w-2xl">
            <p>Software Developer passionate about the ever-evolving realm of Computer science.</p>
            <p>Always learning, always building, always improving</p>
          </div>
        </div>
      </div>
    </header>
  );
}
